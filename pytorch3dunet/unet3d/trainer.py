import os
from numpy.lib.arraysetops import isin
from numpy.lib.ufunclike import isneginf

import torch
import torch.nn as nn
import time
# from tensorboardX import SummaryWriter

# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.datasets.standard_dataset import (
    get_train_val_loaders_dynamic,
    get_test_loaders_dynamic,
)
from pytorch3dunet.unet3d.trainer_utils import _gen_evaluations
from pytorch3dunet.datasets.standard_dataset import (
    _itk_read_array_from_file,
    _itk_read_image_from_file,
)
import pickle
import numpy as np
import SimpleITK as sitk
import h5py
import json

from pytorch3dunet.unet3d.losses import (
    get_loss_criterion, 
    WeightedBCEWithLogitsLoss, 
    DistillationLoss, 
    ContrastiveLoss, 
    pre_contractive_pixel,
)
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from . import utils
from pytorch3dunet.unet3d.trainer_utils import (
    _gen_spatial_weights,
    _get_stage,
    _gen_predictions,
    _gen_num_slices_gt_label_adhoc,
)



logger = get_logger('UNet3DTrainer')

def _create_trainer(config, model, model_prev, is_model_prev_ready, optimizer, lr_scheduler, loss_criterion, 
        loss_criterion_kd, loss_criterion_con, loss_criterion_con_early, loss_criterion_conbr, loss_criterion_con_late,
        is_edge_only_con,
        is_edge_only_con_early,
        is_edge_only_conbr,
        is_edge_only_con_late,
        is_lwf,
        is_kd_mask,
        lambda_kd,
        lambda_con,
        lambda_con_early,
        lambda_conbr,
        lambda_con_late,
        eccv_save_dir,
        eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)


    if resume is not None:        
        raise NotImplementedError()        
    elif pre_trained is not None:
        raise NotImplementedError()        
    else:
        # start training from scratch
        return UNet3DTrainer(
            model=model,
            model_prev=model_prev,
            is_model_prev_ready=is_model_prev_ready,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            loss_criterion_kd=loss_criterion_kd,
            loss_criterion_con=loss_criterion_con,
            loss_criterion_con_early=loss_criterion_con_early,
            loss_criterion_conbr=loss_criterion_conbr,
            loss_criterion_con_late=loss_criterion_con_late,
            is_edge_only_con=is_edge_only_con,
            is_edge_only_con_early=is_edge_only_con_early,
            is_edge_only_conbr=is_edge_only_conbr,
            is_edge_only_con_late=is_edge_only_con_late,
            is_lwf=is_lwf,
            is_kd_mask=is_kd_mask,
            lambda_kd=lambda_kd,
            lambda_con=lambda_con,
            lambda_con_early=lambda_con_early,
            lambda_conbr=lambda_conbr,
            lambda_con_late=lambda_con_late,
            eccv_save_dir=eccv_save_dir,
            eval_criterion=eval_criterion,            
            device=config['device'],
            loaders=loaders,
            **trainer_config
        )


class UNet3DTrainerBuilder:
    @staticmethod
    def build(config, create_loader=False, is_2d=False, included_2d_pl=None, is_conbr_head=True, is_kd_head=True,
        is_kd_loss=False, 
        is_lwf=False,
        is_kd_mask=False,
        lambda_kd=1.0,
        temperature_kd=1.0,         
        is_con_loss=False, 
        lambda_con=1.0,
        temperature_con=1.0, 
        is_edge_only_con=False,
        is_con_early_loss=False, 
        lambda_con_early=1.0,
        temperature_con_early=1.0, 
        is_edge_only_con_early=False,
        is_conbr_loss=False,
        lambda_conbr=1.0,
        temperature_conbr=1.0, 
        is_edge_only_conbr=False,
        is_con_late_loss=False, 
        lambda_con_late=1.0,
        temperature_con_late=1.0,
        is_edge_only_con_late=False,
        eccv_save_dir=None,
    ):
        """
        online: bool, if True, then image is fed into the network one by one
        is_conbr_head: used by vnet model, whether create parallel contrastive head
        is_kd_head: used by vnet model, whether create parallel distillation head
        """
        if is_lwf:
            assert is_kd_loss
            
        if is_kd_loss:
            if not is_lwf:
                assert is_kd_head

        if is_conbr_loss:
            assert is_conbr_head

        assert "model" in config
        config['model']["is_conbr_head"] = is_conbr_head
        config['model']["is_kd_head"] = is_kd_head
        assert "loss" in config
        config["loss"]["is_kd_loss"] = is_kd_loss
        config["loss"]["is_con_loss"] = is_con_loss
        config["loss"]["is_con_early_loss"] = is_con_early_loss
        config["loss"]["is_conbr_loss"] = is_conbr_loss
        config["loss"]["is_con_late_loss"] = is_con_late_loss

        # Create the model
        if is_2d:
            raise NotImplementedError()            
        else:
            model = get_model(config['model'])
            # previous model for knowledge distillation, this is another model class object!
            model_prev = get_model(config['model'])
            # use DataParallel if more than 1 GPU available
            device = config['device']
            if torch.cuda.device_count() > 1 and not device.type == 'cpu':
                model = nn.DataParallel(model)
                model_prev = nn.DataParallel(model_prev)
                logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

            # put the model on GPUs
            logger.info(f"Sending the model to '{config['device']}'")
            model = model.to(device)
            logger.info(f"Sending the model_prev to '{config['device']}'")
            model_prev.testing = True
            model_prev = model_prev.to(device)
            # Log the number of learnable parameters
            logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

            # Create loss criterion
            loss_criterion = get_loss_criterion(config, key_name="loss")
            loss_criterion_kd = None
            loss_criterion_con = None
            loss_criterion_con_early = None
            loss_criterion_conbr = None
            loss_criterion_con_late = None

            # loss criterion for knowledge distillation
            if is_kd_loss:
                print(f"DistillationLoss temperature_kd = {temperature_kd}")
                loss_criterion_kd = DistillationLoss(temperature=temperature_kd)

            if is_con_loss:
                print(f"ContrastiveLoss temperature_con = {temperature_con} is_edge_only_con = {is_edge_only_con}")
                loss_criterion_con = ContrastiveLoss(device=device, temperature=temperature_con)

            if is_con_early_loss:
                print(f"ContrastiveLoss temperature_con_early = {temperature_con_early} is_edge_only_con_early = {is_edge_only_con_early}")
                loss_criterion_con_early = ContrastiveLoss(device=device, temperature=temperature_con_early)

            if is_conbr_loss:
                print(f"ContrastiveLoss temperature_conbr = {temperature_conbr} is_edge_only_conbr = {is_edge_only_conbr}")
                loss_criterion_conbr = ContrastiveLoss(device=device, temperature=temperature_conbr)

            if is_con_late_loss:
                print(f"ContrastiveLoss temperature_con_late = {temperature_con_late} is_edge_only_con_late = {is_edge_only_con_late}")
                loss_criterion_con_late = ContrastiveLoss(device=device, temperature=temperature_con_late)
            
            # Create evaluation metric
            eval_criterion = get_evaluation_metric(config, key_name="eval_metric")

            # Create the optimizer
            # optimizer only created for model, since each round model_prev will be loading model's status
            optimizer = create_optimizer(config['optimizer'], model)

            # Create learning rate adjustment strategy
            lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

        # Create data loaders
        if create_loader:
            raise NotImplementedError()
            loaders = get_train_loaders(config)
        else:
            loaders = None

        # Create model trainer
        trainer = _create_trainer(config, model=model, model_prev=model_prev, is_model_prev_ready=False, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, loss_criterion_kd=loss_criterion_kd, loss_criterion_con=loss_criterion_con, 
                                  loss_criterion_con_early=loss_criterion_con_early, loss_criterion_conbr=loss_criterion_conbr,
                                  loss_criterion_con_late=loss_criterion_con_late,
                                  is_edge_only_con=is_edge_only_con,
                                  is_edge_only_con_early=is_edge_only_con_early,
                                  is_edge_only_conbr=is_edge_only_conbr,
                                  is_edge_only_con_late=is_edge_only_con_late,
                                  is_lwf=is_lwf,
                                  is_kd_mask=is_kd_mask,
                                  lambda_kd=lambda_kd,
                                  lambda_con=lambda_con,
                                  lambda_con_early=lambda_con_early,
                                  lambda_conbr=lambda_conbr,
                                  lambda_con_late=lambda_con_late,
                                  eccv_save_dir=eccv_save_dir,
                                  eval_criterion=eval_criterion, loaders=loaders)

        return trainer

class MultiStageTrainer:
    """wrapper of trainer that supports init train and online train"""
    
    def __init__(
        self, 
        data_root, 
        exp_root, 
        init_train_config, 
        online_train_config, 
        test_config, 
        is_2d, 
        random_seed,        
        random_seed_shuffle_train_set,
        random_seed_propose_annotation,
        n_train1,
        n_train2,
        n_offline_first,
        n_online_first,
        n_test,
        sequential_evaluation_gaps,
        init_train_set, 
        online_train_set,
        test_set,
        data_path_suffix,
        label_path_suffix,
        train_val_p,
        online_legacy_capacity,
        online_legacy_capacity_from_offline,
        online_new_capacity,
        online_new_mcmc_capacity,
        init_max_num_iterations,
        online_max_num_iterations,
        online_annotation_type,
        online_annotation_rounds,
        online_annotation_actions_per_round,
        sample_selection_weights_algo,
        loss_weights_algo,
        proxy_label_weights_algo,
        propose_online_annotation_algo,
        slice_confidence_neib_radius,
        proxy_label_gen_algo, 
        is_ueven_online_iter=False,       
        is_cold_start_interactive_seg=False,
        gen_label_adhoc_only=False,
        model_pred_conf_seed_thres=None,
        use_fixed_warmup_to_guide=False,
        max_cpu_cores=8,
        skip_train_first_annotation_rounds=-1,
        is_save_for_paper=False,
        is_conbr_head=True,
        is_kd_head=True,
        is_kd_loss=False, 
        is_lwf=False,
        is_kd_mask=False,
        lambda_kd=1.0,
        temperature_kd=1.0,         
        is_con_loss=False, 
        lambda_con=1.0,
        temperature_con=1.0, 
        is_edge_only_con=False,
        is_con_early_loss=False, 
        lambda_con_early=1.0,
        temperature_con_early=1.0, 
        is_edge_only_con_early=False,
        is_conbr_loss=False,
        lambda_conbr=1.0,
        temperature_conbr=1.0, 
        is_edge_only_conbr=False,
        is_con_late_loss=False, 
        lambda_con_late=1.0,
        temperature_con_late=1.0,
        is_edge_only_con_late=False,
        eccv_save_dir=None,
    ):
        """
        init_train_config, online_train_config, test_config are three configs
        they define params at each stage
        data_root: root dir for all train/eval data with labels
        exp_root: root dir for all experiment logs and checkpoints
        random_seed_propose_annotation: random seed to control proposed annotation (if there are randomness)
        is_cold_start_interactive_seg: if True, we reset model params after each sample's interactive segmentation
        """
        
        self.is_2d = is_2d
        assert online_annotation_type in ["full_3d", "slice-z", "slice-y", "slice-x", "slice-zy", "slice-zx", "slice-yx", "slice-zyx", "points"]
        self.online_annotation_type = online_annotation_type
        self.included_2d_pl = None
        if self.is_2d and "slice-" in self.online_annotation_type:
            self.included_2d_pl = [x for x in self.online_annotation_type.split("-")[1]]

        self.online_annotation_rounds = online_annotation_rounds
        self.online_annotation_actions_per_round = online_annotation_actions_per_round

        if self.online_annotation_type == "full_3d" and online_train_set != "none":
            assert self.online_annotation_rounds == 1 and self.online_annotation_actions_per_round == 1

        self.proxy_label_gen_algo = proxy_label_gen_algo

        if self.online_annotation_type == "points":
            assert self.proxy_label_gen_algo == "fastgc_3d"

        if self.online_annotation_type.startswith("slice-"):
            assert not self.proxy_label_gen_algo == "fastgc_3d"

        self.is_conbr_head=is_conbr_head
        self.is_kd_head=is_kd_head
        self.is_kd_loss=is_kd_loss
        self.is_lwf=is_lwf
        self.is_kd_mask=is_kd_mask
        self.lambda_kd=lambda_kd
        self.temperature_kd=temperature_kd
        self.is_con_loss=is_con_loss
        self.lambda_con=lambda_con
        self.temperature_con=temperature_con
        self.is_edge_only_con=is_edge_only_con
        self.is_con_early_loss=is_con_early_loss
        self.lambda_con_early=lambda_con_early
        self.temperature_con_early=temperature_con_early
        self.is_edge_only_con_early=is_edge_only_con_early
        self.is_conbr_loss=is_conbr_loss
        self.lambda_conbr=lambda_conbr
        self.temperature_conbr=temperature_conbr
        self.is_edge_only_conbr=is_edge_only_conbr
        self.is_con_late_loss=is_con_late_loss
        self.lambda_con_late=lambda_con_late
        self.temperature_con_late=temperature_con_late
        self.is_edge_only_con_late=is_edge_only_con_late
        self.eccv_save_dir = eccv_save_dir
                
        self.init_trainer = UNet3DTrainerBuilder.build(init_train_config, create_loader=False, is_2d=is_2d, included_2d_pl=self.included_2d_pl,
            is_conbr_head=self.is_conbr_head, is_kd_head=self.is_kd_head,
            is_kd_loss=self.is_kd_loss,
            is_lwf=self.is_lwf,
            is_kd_mask=self.is_kd_mask,
            lambda_kd=self.lambda_kd,
            temperature_kd=self.temperature_kd,
            is_con_loss=self.is_con_loss,
            lambda_con=self.lambda_con,
            temperature_con=self.temperature_con,
            is_edge_only_con=self.is_edge_only_con,
            is_con_early_loss=self.is_con_early_loss, 
            lambda_con_early=self.lambda_con_early,
            temperature_con_early=self.temperature_con_early, 
            is_edge_only_con_early=self.is_edge_only_con_early,
            is_conbr_loss=self.is_conbr_loss,
            lambda_conbr=self.lambda_conbr,
            temperature_conbr=self.temperature_conbr,
            is_edge_only_conbr=self.is_edge_only_conbr,
            is_con_late_loss=self.is_con_late_loss,
            lambda_con_late=self.lambda_con_late,
            temperature_con_late=self.temperature_con_late,
            is_edge_only_con_late=self.is_edge_only_con_late,
            eccv_save_dir=self.eccv_save_dir,
        )
        self.online_trainer = UNet3DTrainerBuilder.build(online_train_config, create_loader=False, is_2d=is_2d, included_2d_pl=self.included_2d_pl,
            is_conbr_head=self.is_conbr_head, is_kd_head=self.is_kd_head,
            is_kd_loss=self.is_kd_loss,
            is_lwf=self.is_lwf,
            is_kd_mask=self.is_kd_mask,
            lambda_kd=self.lambda_kd,
            temperature_kd=self.temperature_kd,
            is_con_loss=self.is_con_loss,
            lambda_con=self.lambda_con,
            temperature_con=self.temperature_con,
            is_edge_only_con=self.is_edge_only_con,
            is_con_early_loss=self.is_con_early_loss, 
            lambda_con_early=self.lambda_con_early,
            temperature_con_early=self.temperature_con_early, 
            is_edge_only_con_early=self.is_edge_only_con_early,
            is_conbr_loss=self.is_conbr_loss,
            lambda_conbr=self.lambda_conbr,
            temperature_conbr=self.temperature_conbr,
            is_edge_only_conbr=self.is_edge_only_conbr,
            is_con_late_loss=self.is_con_late_loss,
            lambda_con_late=self.lambda_con_late,
            temperature_con_late=self.temperature_con_late,
            is_edge_only_con_late=self.is_edge_only_con_late,
            eccv_save_dir=self.eccv_save_dir,
        )

        self.init_train_config = init_train_config
        self.online_train_config = online_train_config
        self.test_config = test_config
        self.model = self.init_trainer.model
        self.data_root = data_root

        self.exp_root = exp_root
        self.is_save_for_paper = is_save_for_paper
        # if save for paper, then we save intermediate images, masks, weights, slices into a pickle file and retriev it when displaying in the paper!
        
        self.random_seed = random_seed
        self.random_seed_propose_annotation = random_seed_propose_annotation

        self.train_val_p=train_val_p
        self.online_legacy_capacity=online_legacy_capacity
        self.online_legacy_capacity_from_offline=online_legacy_capacity_from_offline
        self.online_new_capacity=online_new_capacity
        self.online_new_mcmc_capacity=online_new_mcmc_capacity

        self.init_max_num_iterations = init_max_num_iterations
        self.online_max_num_iterations = online_max_num_iterations

        

        self.sample_selection_weights_algo = sample_selection_weights_algo
        self.loss_weights_algo = loss_weights_algo
        self.proxy_label_weights_algo = proxy_label_weights_algo
        self.propose_online_annotation_algo = propose_online_annotation_algo
        self.slice_confidence_neib_radius = slice_confidence_neib_radius        
        
        self.is_ueven_online_iter = is_ueven_online_iter
        self.is_cold_start_interactive_seg = is_cold_start_interactive_seg
        self.gen_label_adhoc_only = gen_label_adhoc_only

        self.model_pred_conf_seed_thres = model_pred_conf_seed_thres

        self.use_fixed_warmup_to_guide = use_fixed_warmup_to_guide
        
        self.max_cpu_cores = max_cpu_cores

        self.skip_train_first_annotation_rounds = skip_train_first_annotation_rounds

        # read meta
        with open(os.path.join(self.data_root, "sample_id_dict.json"), 'r') as f:
            sample_id_dict = json.load(f)

        # split data
        np.random.seed(random_seed)
        all_sample_ids = sorted(list(sample_id_dict.keys()))
        _ = np.random.shuffle(all_sample_ids)

        all_sample_ids_train = all_sample_ids[0 : n_train1 + n_train2]
        if n_test > 0:
            all_sample_ids_test = all_sample_ids[n_train1 + n_train2 : n_train1 + n_train2 + n_test]
        # possible shuffling of initial_sample_id_queue + online_sample_id_queue to create statistically trusted experiments
        if random_seed_shuffle_train_set >= 0:
            np.random.seed(random_seed_shuffle_train_set)
            _ = np.random.shuffle(all_sample_ids_train)            

        train1_ids = ["sample_{}".format(x) for x in all_sample_ids_train[0 : n_train1]]
        train2_ids = ["sample_{}".format(x) for x in all_sample_ids_train[n_train1 : n_train1 + n_train2]]
        if n_test > 0:
            test_ids = ["sample_{}".format(x) for x in all_sample_ids_test]
        else:
            test_ids = []

        assert len(train1_ids) == n_train1
        assert len(train2_ids) == n_train2
        
        if n_test > 0:
            assert len(test_ids) == n_test
        
        self.n_test = n_test
        self.sequential_evaluation_gaps = sequential_evaluation_gaps

        train_ids = {
            "train1": train1_ids,
            "train2": train2_ids,
            "train12": train1_ids + train2_ids,
            "test": test_ids,
            "train12_test": train1_ids + train2_ids + test_ids,
        }
    
        initial_sample_id_queue = train_ids.get(init_train_set, [])
        online_sample_id_queue = train_ids.get(online_train_set, [])
        eval_sample_id_queue = train_ids.get(test_set, [])

        if n_offline_first > 0 and n_online_first > 0:
            raise RuntimeError(f"only one of n_offline_first and n_online_first could be positive!")

        if n_offline_first >= 0:
            initial_sample_id_queue = train_ids.get(init_train_set, [])[:n_offline_first]
            if n_test == -1:
                # use rest of initial sample_id_queue as testing set!
                eval_sample_id_queue = train_ids.get(init_train_set, [])[n_offline_first:]

        if n_online_first >= 0:
            online_sample_id_queue = train_ids.get(online_train_set, [])[:n_online_first]

        assert len(initial_sample_id_queue) or len(online_sample_id_queue) or len(eval_sample_id_queue)
        
        self.initial_data_paths = [os.path.join(self.data_root, x + data_path_suffix) for x in initial_sample_id_queue]
        self.initial_label_paths = [os.path.join(self.data_root, x + label_path_suffix) for x in initial_sample_id_queue]

        self.online_data_paths = [os.path.join(self.data_root, x + data_path_suffix) for x in online_sample_id_queue]
        self.online_label_paths = [os.path.join(self.data_root, x + label_path_suffix) for x in online_sample_id_queue]

        self.eval_data_paths = [os.path.join(self.data_root, x + data_path_suffix) for x in eval_sample_id_queue]
        self.eval_label_paths = [os.path.join(self.data_root, x + label_path_suffix) for x in eval_sample_id_queue]
        
        self.SAVE_FOR_PAPER_PATH = None
        if self.is_save_for_paper:
            self.SAVE_FOR_PAPER_PATH = "{}/paper".format(self.exp_root)
            if not os.path.exists(self.SAVE_FOR_PAPER_PATH):
                os.makedirs(self.SAVE_FOR_PAPER_PATH)


    def infer(self, model_checkpoint, save_for_paper_path):
        # if we apply sequential test, then we should avoiding missing the results!
        self.online_trainer.model.load_state_dict(torch.load(model_checkpoint, map_location=self.online_trainer.device))
        gt_test_score = self.online_trainer.test(
            test_config=self.test_config,
            stage="infer",
            eval_data_paths=self.eval_data_paths,
            eval_label_paths=self.eval_label_paths,
            is_2d=False,
            included_2d_pl=None,
            included_2d_slices=None,
            global_num_updates=0,
            is_save_for_paper=True,
            save_for_paper_path=save_for_paper_path,
        )

    def _train_single_stage(
        self,
        trainer,
        train_config,
        new_data_path=None,
        new_label_path=None,
        legacy_dataset=None,
        legacy_dataset_from_offline=None,
        global_num_updates=0,
        stage="dynamic-init",
        is_online_stage=False,
        included_2d_slices=None,
        len_option="min",
        sample_selection_weights=None,
        loss_weights=None,
        is_reset_lr=False,
        i_effort=0,
        random_seed=None,
        is_model_prev_ready=False,
    ):
        """init training on batch samples as warming up stage
            train_config: train_config
            initial_data_paths: List[nrrd]
            initial_label_paths: List[nrrd]

            included_2d_slices: to be determined by practical training settings
            is_model_prev_ready: whether trainer.model_prev is ready to make inference
        """
        trainer.is_model_prev_ready = is_model_prev_ready
        if is_reset_lr:
            # reset optimizer and scheduler
            trainer._reset_optimizer_and_scheduler(
                train_config, 
                is_2d=self.is_2d,
                included_2d_pl=self.included_2d_pl,
            )
        # get data loaders
        loaders, legacy_dataset, train_dataset, val_dataset = get_train_val_loaders_dynamic(
            config=train_config,
            legacy_dataset=legacy_dataset, 
            legacy_dataset_from_offline=legacy_dataset_from_offline,
            initial_data_paths=None if is_online_stage else self.initial_data_paths, 
            initial_label_paths=None if is_online_stage else self.initial_label_paths,
            new_data_path=new_data_path, 
            new_label_path=new_label_path,
            new_weighted_label_path=new_data_path,
            online_legacy_capacity=self.online_legacy_capacity if is_online_stage else None,
            online_legacy_capacity_from_offline=self.online_legacy_capacity_from_offline if is_online_stage else None,
            online_new_capacity=self.online_new_capacity if is_online_stage else None,
            online_new_mcmc_capacity=self.online_new_mcmc_capacity if is_online_stage else None,
            train_val_p=self.train_val_p,
            random_seed=random_seed,
            is_2d=self.is_2d,
            included_2d_pl=self.included_2d_pl,
            included_2d_slices=included_2d_slices,
            len_option=len_option,
            sample_selection_weights=sample_selection_weights,
            loss_weights=loss_weights,
        )

        # initial training
        trainer.train_loaders, trainer.val_loaders = loaders["train"], loaders["val"]
        trainer.n_test = self.n_test
        n_val_patches_ = len(val_dataset) if val_dataset is not None else 0
        logger.info(f'TRAIN - {stage} - iter{global_num_updates}, train {len(trainer.train_loaders)} batches, {len(train_dataset)} patches, val {len(trainer.val_loaders)} batches, {n_val_patches_} patches')
        
        # print train sample ids
        curr_train_sample_ids_ = []
        # for _, _, _, _, sample_id_int in train_dataset:
        #     curr_train_sample_ids_.append(sample_id_int)
        # curr_train_sample_ids_ = sorted(curr_train_sample_ids_)
        logger.info(f'TRAIN-SAMPLE-IDS - {stage} - iter{global_num_updates}, train {len(curr_train_sample_ids_)} samples, sample_ids: {curr_train_sample_ids_}')

        num_epochs = 0
        num_updates = 0
        is_stop_per_pl = None

        if is_online_stage:
            if self.is_ueven_online_iter:
                if self.skip_train_first_annotation_rounds > 0:
                    if i_effort < self.skip_train_first_annotation_rounds:
                        max_num_updates = 0
                    else:
                        max_num_updates = self.online_max_num_iterations
                else:
                    max_num_updates = int(float(i_effort + 1) / self.online_annotation_rounds * self.online_max_num_iterations)
            else:
                max_num_updates = self.online_max_num_iterations
        else:
            max_num_updates = self.init_max_num_iterations
        if max_num_updates < 1:
            print(f"skip_train_one_epoch")
            return legacy_dataset, global_num_updates
        while True:
            # train for one epoch
            should_terminate, is_stop_per_pl, global_num_updates, num_updates = trainer.train_one_epoch(
                global_num_updates=global_num_updates,
                num_updates=num_updates,
                max_num_updates=max_num_updates,
                num_epochs=num_epochs,
                test_config=self.test_config,
                eval_data_paths=self.eval_data_paths,
                eval_label_paths=self.eval_label_paths,
                stage=stage,
                is_2d=self.is_2d,
                included_2d_pl=self.included_2d_pl,
                included_2d_slices=included_2d_slices,
                is_stop_per_pl=is_stop_per_pl,
                is_save_for_paper=self.is_save_for_paper,
                save_for_paper_path=self.SAVE_FOR_PAPER_PATH,
            )
            if should_terminate:
                logger.info(f'TRAIN - {stage} - iter{global_num_updates}, stopped')
                break
            num_epochs += 1
        return legacy_dataset, global_num_updates
        
    def fit_dynamic_one_round(
        self,
        trainer,
        new_data_path,
        new_label_path,
        proxy_label,
        existing_label_slices,
        legacy_dataset,
        legacy_dataset_from_offline,
        global_num_updates,
        total_samples_seen,
        new_sample_id,
        i_effort,        
        fixed_previous_sample_model_curr_pred,
        curr_pred,
        existing_fastgc_seed_indices=None,
        existing_fastgc_prediction=None,
        prev_contributor_pred_dict=None,
    ):
        """fit one round of dynamic training
        1. given new sample
        2. generate predictions
        3. propose new annotations
        4. propagate new annotations
        5. evaluate predictions
        6. generate spatial weights
        7. train against new propagated annotations
        """
         # stage_contributor = stage + "-ie-{}-{}".format(i_effort, NUM_TOTAL_SLICE_LABEL_EFFORTS)        
        stage_contributor = _get_stage(stype="dynamic-online", total_samples_seen=total_samples_seen,
            sample_id=new_sample_id, i_effort=i_effort, total_effort=self.online_annotation_rounds, 
            existing_label_slices=existing_label_slices, existing_fastgc_seeds=existing_fastgc_seed_indices)
        
        initial_fastgc_seed_map = None
        proxy_label_dict = {}
        if self.online_annotation_type == "points":
            raise NotImplementedError()            
        elif "slice-" in self.online_annotation_type:
            raise NotImplementedError()            
        else:
            proxy_label = None
        

        stage_contributor = _get_stage(stype="dynamic-online", total_samples_seen=total_samples_seen,
                sample_id=new_sample_id, i_effort=i_effort, total_effort=self.online_annotation_rounds, 
                existing_label_slices=existing_label_slices, existing_fastgc_seeds=existing_fastgc_seed_indices)
        logger.info(f"GEN_contributor_PRED - {stage_contributor} - iter{global_num_updates}")
        score = _gen_evaluations(
            eval_criterion=trainer.eval_criterion,
            label_path=new_label_path,
            pred_path=curr_pred,
            propagate_pred_path=proxy_label,
            existing_label_slices=existing_label_slices,
            is_2d=self.is_2d,
        )

        if initial_fastgc_seed_map is not None:
            raise NotImplementedError()
        else:
            logger.info(f"EVAL_SAMPLE_SCORE - {stage_contributor} - iter{global_num_updates}, sample_eval_score = {score}")
        
        if self.proxy_label_weights_algo == "sigmoid":
            proxy_label_weights_algo = np.exp(total_samples_seen) / (np.exp(total_samples_seen) + 1) if total_samples_seen > 1 else None
        elif self.proxy_label_weights_algo == "sigmoid-onlineonly":
            if len(self.initial_data_paths) > 0:
                x = total_samples_seen - len(self.initial_data_paths) + 1
                proxy_label_weights_algo = np.exp(x) / (np.exp(x) + 1)
            else:
                proxy_label_weights_algo = np.exp(total_samples_seen) / (np.exp(total_samples_seen) + 1) if total_samples_seen > 1 else None
        elif self.proxy_label_weights_algo == "sigmoid-scaled":
            n_total_ = float(len(self.initial_data_paths) + len(self.online_data_paths))
            if total_samples_seen > 1:
                x_ = 4 * (total_samples_seen / n_total_ - 0.5)
                proxy_label_weights_algo = np.exp(x_) / (np.exp(x_) + 1)
            else:
                proxy_label_weights_algo = None
        elif self.proxy_label_weights_algo == "inv-sigmoid-scaled":
            n_total_ = float(len(self.initial_data_paths) + len(self.online_data_paths))
            if total_samples_seen > 1:
                x_ = 4 * (total_samples_seen / n_total_ - 0.5)
                proxy_label_weights_algo = 1.0 - np.exp(x_) / (np.exp(x_) + 1)
            else:
                proxy_label_weights_algo = None
        else:
            proxy_label_weights_algo = self.proxy_label_weights_algo if total_samples_seen > 1 else None

        # if proxy_label is None, we use true labels
        sample_selection_weights, loss_weights, proxy_label_weights, weighted_proxy_label = _gen_spatial_weights(
            image_path=new_data_path,
            pred_path=curr_pred,
            fixed_previous_sample_model_curr_pred=fixed_previous_sample_model_curr_pred,
            label_path=new_label_path,
            propagate_pred_path=proxy_label,
            existing_label_slices=existing_label_slices,
            output_spatial_prior_path=None,
            sample_selection_weights_algo=self.sample_selection_weights_algo if total_samples_seen > 1 else None,
            loss_weights_algo=self.loss_weights_algo,
            proxy_label_weights_algo=proxy_label_weights_algo,
            slice_confidence_neib_radius=self.slice_confidence_neib_radius,
            save_file=False,
            is_2d=self.is_2d,
            is_use_gt=True if proxy_label is None else False,
            existing_fastgc_seed_indices=existing_fastgc_seed_indices,
            initial_fastgc_seed_map=initial_fastgc_seed_map,
        )
        logger.info(f"GEN_SPATIAL_PRIOR - {stage_contributor} - iter{global_num_updates}")
        

        # note that is_model_prev_ready is True only if we have seen at least one sample from the stream!
        legacy_dataset_, global_num_updates = self._train_single_stage(
            trainer,
            self.online_train_config,
            new_data_path=new_data_path,
            new_label_path=weighted_proxy_label,
            legacy_dataset=legacy_dataset,
            legacy_dataset_from_offline=legacy_dataset_from_offline,
            global_num_updates=global_num_updates,
            stage=stage_contributor,
            is_online_stage=True,
            included_2d_slices=None,
            len_option="min",
            sample_selection_weights=sample_selection_weights,
            loss_weights=loss_weights,
            is_reset_lr=False,
            i_effort=i_effort,
            random_seed=self.random_seed + (total_samples_seen - len(self.initial_data_paths) + 1) * 2 * self.online_annotation_rounds + i_effort,
            is_model_prev_ready=True if total_samples_seen > 1 else False,
        )
        
        save_for_paper_additional_info = {}
        if self.is_save_for_paper:
            save_for_paper_additional_info = {
                "sample_selection_weights": sample_selection_weights, 
                "loss_weights": loss_weights,
                "proxy_label_weights": proxy_label_weights,
                "weighted_proxy_label": weighted_proxy_label,
                "fixed_previous_sample_model_curr_pred": fixed_previous_sample_model_curr_pred,
                "sample_selection_weights_algo": self.sample_selection_weights_algo,
                "loss_weights_algo": self.loss_weights_algo,
                "proxy_label_weights_algo": proxy_label_weights_algo,
                "slice_confidence_neib_radius": self.slice_confidence_neib_radius,
            }

        return legacy_dataset_, global_num_updates, proxy_label, existing_label_slices, existing_fastgc_seed_indices, existing_fastgc_prediction, proxy_label_dict, stage_contributor, save_for_paper_additional_info

    def fit_dynamic(self):
        # init stage
        total_samples_seen = 0
        global_num_updates = 0
        legacy_dataset = None
        legacy_dataset_from_offline = None
        if len(self.initial_data_paths) and len(self.initial_data_paths):
            total_samples_seen += len(self.initial_data_paths)
            # during initial batch training, we should set is_model_prev_ready = False
            legacy_dataset_from_offline, global_num_updates = self._train_single_stage(
                self.init_trainer,
                self.init_train_config,
                new_data_path=None,
                new_label_path=None,
                legacy_dataset=legacy_dataset_from_offline,
                global_num_updates=global_num_updates,
                stage=_get_stage("dynamic-init", total_samples_seen),
                is_online_stage=False,
                included_2d_slices=None,
                len_option="min",
                sample_selection_weights=None,
                loss_weights=None,
                is_reset_lr=True,
                i_effort=None,
                random_seed=self.random_seed,
                is_model_prev_ready=False,
            )
            # transfer model and learning state from init trainer to online trainer
            if isinstance(self.init_trainer.model, dict):
                for k in self.init_trainer.model.keys():
                    self.online_trainer.model[k].load_state_dict(self.init_trainer.model[k].state_dict())
            else:
                self.online_trainer.model.load_state_dict(self.init_trainer.model.state_dict())

            if isinstance(self.init_trainer.optimizer, dict):
                for k in self.init_trainer.optimizer.keys():
                    self.online_trainer.optimizer[k].load_state_dict(self.init_trainer.optimizer[k].state_dict())
            else:
                self.online_trainer.optimizer.load_state_dict(self.init_trainer.optimizer.state_dict())

            if isinstance(self.init_trainer.scheduler, dict):
                for k in self.init_trainer.scheduler.keys():
                    self.online_trainer.scheduler[k].load_state_dict(self.init_trainer.scheduler[k].state_dict())
            else:
                self.online_trainer.scheduler.load_state_dict(self.init_trainer.scheduler.state_dict())
        
        # online stage
        legacy_dataset = legacy_dataset_from_offline
        if len(self.online_data_paths) and len(self.online_label_paths):            
            
            fixed_previous_sample_model_curr_pred = None
            fixed_previous_sample_model_legacy_dataset = None
            for ii, vv in enumerate(zip(self.online_data_paths, self.online_label_paths)):
                new_data_path, new_label_path = vv
                label_adhoc = _gen_num_slices_gt_label_adhoc(new_label_path)
                logger.info(f"EVAL_LABEL_ADHOC - sn{ii + 1} - filename_{new_data_path}, label_adhoc = {label_adhoc}")
                if self.gen_label_adhoc_only:
                    continue

                if self.is_cold_start_interactive_seg:
                    # reset everything so the model is untrained at the beginning!
                    # transfer model and learning state from init trainer to online trainer
                    if isinstance(self.init_trainer.model, dict):
                        for k in self.init_trainer.model.keys():
                            self.online_trainer.model[k].load_state_dict(self.init_trainer.model[k].state_dict())
                    else:
                        self.online_trainer.model.load_state_dict(self.init_trainer.model.state_dict())

                    if isinstance(self.init_trainer.optimizer, dict):
                        for k in self.init_trainer.optimizer.keys():
                            self.online_trainer.optimizer[k].load_state_dict(self.init_trainer.optimizer[k].state_dict())
                    else:
                        self.online_trainer.optimizer.load_state_dict(self.init_trainer.optimizer.state_dict())

                    if isinstance(self.init_trainer.scheduler, dict):
                        for k in self.init_trainer.scheduler.keys():
                            self.online_trainer.scheduler[k].load_state_dict(self.init_trainer.scheduler[k].state_dict())
                    else:
                        self.online_trainer.scheduler.load_state_dict(self.init_trainer.scheduler.state_dict())

                
                if self.n_test == -1:
                    if (ii + 1) % self.sequential_evaluation_gaps == 0:
                        # we use the rest of training samples (those not participated in training) as evaluation
                        self.eval_data_paths = self.online_data_paths[(ii+1) : ]
                        self.eval_label_paths = self.online_label_paths[(ii+1) : ]
                    else:
                        self.eval_data_paths = []
                        self.eval_label_paths = []
                total_samples_seen += 1
                if self.online_annotation_type == "points":     
                    raise NotImplementedError()                         
                elif "slice-" in self.online_annotation_type:
                    raise NotImplementedError()                    
                elif self.online_annotation_type == "full_3d":
                    existing_label_slices = None
                    existing_fastgc_seed_indices = set()
                    existing_fastgc_prediction = None
                else:
                    raise NotImplementedError()
                proxy_label = None
                prev_contributor_pred_dict = None

                save_for_paper_dict = {}
                if self.is_save_for_paper:
                    save_for_paper_dict["image_array"] = _itk_read_array_from_file(new_data_path)  
                    save_for_paper_dict["image_path"] = new_data_path
                    # save ground truth label array
                    save_for_paper_dict["gt_array"] = _itk_read_array_from_file(new_label_path) 
                    save_for_paper_dict["gt_path"] = new_label_path
                for i_effort in range(self.online_annotation_rounds):
                    curr_pred = _gen_predictions(
                        model=self.online_trainer.model,
                        test_config=self.test_config,
                        MODEL_PRED_DIR=None,
                        new_data_path=new_data_path,
                        output_pred_path=None,
                        save_file=False,
                        is_2d=self.is_2d,
                        included_2d_pl=self.included_2d_pl,
                        included_2d_slices=None,
                    )
                    if i_effort == 0:
                        fixed_previous_sample_model_curr_pred = curr_pred
                        fixed_previous_sample_model_legacy_dataset = legacy_dataset
                        # current the setting: we give model_prev the last round of model state, but do not change model_prev during rounds of human interactions
                        self.online_trainer.model_prev.load_state_dict(self.online_trainer.model.state_dict())
                        self.online_trainer.model_prev.eval()
                        self.online_trainer.model_prev.testing = True
                        
                    legacy_dataset, global_num_updates, proxy_label, existing_label_slices, existing_fastgc_seed_indices, existing_fastgc_prediction, prev_contributor_pred_dict, stage_contributor, save_for_paper_additional_info = self.fit_dynamic_one_round(
                        self.online_trainer,
                        new_data_path=new_data_path,
                        new_label_path=new_label_path,
                        proxy_label=proxy_label,
                        existing_label_slices=existing_label_slices,
                        legacy_dataset=fixed_previous_sample_model_legacy_dataset,
                        legacy_dataset_from_offline=legacy_dataset_from_offline,
                        global_num_updates=global_num_updates,
                        total_samples_seen=total_samples_seen,
                        new_sample_id=new_data_path.split("/")[-1].split(".")[0],
                        i_effort=i_effort,
                        fixed_previous_sample_model_curr_pred=fixed_previous_sample_model_curr_pred,
                        curr_pred=curr_pred,
                        existing_fastgc_seed_indices=existing_fastgc_seed_indices,
                        existing_fastgc_prediction=existing_fastgc_prediction,
                        prev_contributor_pred_dict=prev_contributor_pred_dict,
                    )
                    # save for paper:
                    if self.is_save_for_paper:
                        # save curr_pred
                        save_for_paper_dict["curr_pred"] = curr_pred
                        # save proxy label array
                        save_for_paper_dict["proxy_label"] = proxy_label
                        # save prev_contributor_pred_dict
                        save_for_paper_dict["prev_contributor_pred_dict"] = prev_contributor_pred_dict
                        # save existing_label_slices
                        save_for_paper_dict["existing_label_slices"] = existing_label_slices
                        # save save_for_paper_additional_info
                        save_for_paper_dict["save_for_paper_additional_info"] = save_for_paper_additional_info

                        pickle_file_path = os.path.join(self.SAVE_FOR_PAPER_PATH, stage_contributor + ".pickle")
                        with open(pickle_file_path, 'wb') as handle:
                            pickle.dump(save_for_paper_dict, handle)
                        logger.info(f"saved save_for_paper_dict to {pickle_file_path}")

class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        max_validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_epoch (int): useful when loading the model from the checkpoint
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)

        is_model_prev_ready: bool, whether model_prev is ready to make inference, meaning non-empty model state
        loss_criterion_kd: loss criterion for knowledge distillation
        loss_criterion_con: con loss applied on the semantic layer
        loss_criterion_con_early: the con loss applied on earlyer (higher resolution) conv layers
        loss_criterion_conbr: the con branched loss applied to semantic layer
        loss_criterion_con_late: the con loss applied on later conv layers
    """

    def __init__(self, model, model_prev, is_model_prev_ready, optimizer, lr_scheduler, loss_criterion, loss_criterion_kd, loss_criterion_con,
                loss_criterion_con_early, loss_criterion_conbr, loss_criterion_con_late, 
                is_edge_only_con,
                is_edge_only_con_early,
                is_edge_only_conbr,
                is_edge_only_con_late,
                is_lwf,
                is_kd_mask,
                lambda_kd,
                lambda_con,
                lambda_con_early,
                lambda_conbr,
                lambda_con_late,
                eccv_save_dir, 
                eval_criterion, device, loaders,
                validate_after_iters=100, test_after_iters=100, log_after_iters=100,
                max_validate_iters=None,
                eval_score_higher_is_better=True, best_eval_score=None,
                skip_train_validation=False, **kwargs):

        self.model = model
        self.model_prev = model_prev
        self.is_model_prev_ready = is_model_prev_ready
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.loss_criterion_kd = loss_criterion_kd
        self.loss_criterion_con = loss_criterion_con
        self.loss_criterion_con_early = loss_criterion_con_early
        self.loss_criterion_conbr = loss_criterion_conbr
        self.loss_criterion_con_late = loss_criterion_con_late
        self.is_edge_only_con = is_edge_only_con
        self.is_edge_only_con_early = is_edge_only_con_early
        self.is_edge_only_conbr = is_edge_only_conbr
        self.is_edge_only_con_late = is_edge_only_con_late
        self.eccv_save_dir = eccv_save_dir

        self.is_lwf=is_lwf
        self.is_kd_mask=is_kd_mask
        self.lambda_kd=lambda_kd
        self.lambda_con=lambda_con
        self.lambda_con_early=lambda_con_early
        self.lambda_conbr=lambda_conbr
        self.lambda_con_late=lambda_con_late

        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.validate_after_iters = validate_after_iters
        self.test_after_iters = test_after_iters
        self.log_after_iters = log_after_iters
        self.max_validate_iters = max_validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.n_test = None

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            if isinstance(self.model, dict):
                # 2D cases
                if eval_score_higher_is_better:
                    self.best_eval_score = {k: float('-inf') for k in self.model.keys()}
                else:
                    self.best_eval_score = {k: float('+inf') for k in self.model.keys()}
            else:
                # initialize the best_eval_score
                if eval_score_higher_is_better:
                    self.best_eval_score = float('-inf')
                else:
                    self.best_eval_score = float('+inf')

        self.skip_train_validation = skip_train_validation
    
    def _reset_optimizer_and_scheduler(self, train_config, is_2d=False, included_2d_pl=None):
        lr = train_config["optimizer"]["lr"]
        if is_2d:
            raise NotImplementedError()            
        else:
            if hasattr(self.scheduler, "_reset"):
                self.scheduler._reset()
            self.optimizer.param_groups[0]['lr'] = lr
            logger.info(f"RESET - scheduler and lr = {lr}")

    def train_one_epoch(
        self,        
        global_num_updates,
        num_updates,
        max_num_updates,
        num_epochs,
        test_config,
        eval_data_paths,
        eval_label_paths,
        stage="init_training",
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        is_stop_per_pl=None,
        is_save_for_paper=False,
        save_for_paper_path=None,
    ):
        """Trains the model for 1 epoch.

        Here self.model_prev is holding the states from last STAGE (before accepting the current sample in the stream)

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        if is_2d:
            raise NotImplementedError()
        else:
            train_losses = utils.RunningAverage()
            train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        if is_2d:
            raise NotImplementedError()
        else:
            self.model.train()
            self.model.testing = False

        for t in self.train_loaders:
            if is_2d:                
                raise NotImplementedError()                
            else:
                # if hasattr(self.model, "out_tr_kd"):
                #     logger.info(f"before backward out_tr_target sum = {self.model.out_tr_target.conv1.weight.data.sum().item()}")
                #     logger.info(f"before backward out_tr_kd sum = {self.model.out_tr_kd.conv1.weight.data.sum().item()}")
                input, target, _, weight, _ = self._split_training_batch(t)
                # print("|DEBUG target.size() = {}".format(target.size()))

                output, loss, loss_seg = self._forward_pass(input, target, weight=weight, pos_weight=None, is_2d=is_2d, plane_key=None)
                # output, loss = self._forward_pass(input, target, weight=weight, pos_weight=torch.tensor(10.0), is_2d=is_2d, plane_key=None)
                # logger.info(f"DEBUG_W || '{self.model.compress5[0].weight.cpu().detach().numpy()[0,0,0,0,0]}'")

                train_losses.update(loss_seg.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                    # logger.info(f"TRAIN - {stage} - iter{global_num_updates} - scheduler.last_epoch: {self.scheduler.last_epoch}, optm_lr: '{self.optimizer.param_groups[0]['lr']}'")
                # if hasattr(self.model, "out_tr_kd"):
                #     logger.info(f"after backward out_tr_target sum = {self.model.out_tr_target.conv1.weight.data.sum().item()}")
                #     logger.info(f"after backward out_tr_kd sum = {self.model.out_tr_kd.conv1.weight.data.sum().item()}")
            # if global_num_updates % self.log_after_iters == 0:
            if num_updates + 1 > max_num_updates:
                # we log once per epoch
                if is_2d:
                    raise NotImplementedError()
                else:
                    # if model contains final_activation layer for normalizing logits apply it, otherwise both
                    # the evaluation metric as well as images in tensorboard will be incorrectly computed
                    if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                        output = self.model.final_activation(output)

                    # compute eval criterion
                    if not self.skip_train_validation:
                        eval_score = self.eval_criterion(output, target)
                        if eval_score.item() > 1:
                            raise ValueError(f"TRAIN - {stage} - iter{global_num_updates} How could DICE {eval_score.item()} greater than 1? Entering debugging mode")
                        train_eval_scores.update(eval_score.item(), self._batch_size(input))

                    curr_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"""TRAIN - {stage} - iter{global_num_updates}, num_updates: {num_updates}/{max_num_updates},\
                        num_epochs: {num_epochs}, lr: {curr_lr}, scheduler: {self.scheduler.state_dict()}, \
                        train_loss: {np.round(loss_seg.item(), 2)}, avg_train_loss: {train_losses.avg}, train_score: {np.round(eval_score.item(), 2)},\
                        avg_train_score: {train_eval_scores.avg}""")

            # if len(self.val_loaders) > 0 and global_num_updates % self.validate_after_iters == 0:
            if len(self.val_loaders) > 0 and num_updates + 1 > max_num_updates:
                # we log each epoch once
                if is_2d:
                    raise NotImplementedError()
                else:
                    # set the model in eval mode
                    self.model.eval()
                    # evaluate on validation set
                    val_loss, val_score = self.validate()
                    # set the model back to training mode
                    self.model.train()

                    # adjust learning rate if necessary
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_score)
                    # log current learning rate in tensorboard
                    # self._log_lr()
                    # remember best validation metric
                    is_best = self._is_best_eval_score(val_score)
                    
                    logger.info(f"""VAL - {stage} - iter{global_num_updates}, num_updates: {num_updates}/{max_num_updates}, \
                        num_epochs: {num_epochs}, lr: {curr_lr}, \
                        val_loss: {val_loss}, avg_val_loss: {val_loss}, val_score: {val_score}, \
                        avg_val_score: {val_score}, best_val_score: {self.best_eval_score}, is_new_best: {is_best}, last_save_path: none, \
                        best_save_path: none""")

            # if global_num_updates % self.test_after_iters == 0 or (self.n_test == -1 and "online" in stage):
            # if global_num_updates % self.test_after_iters == 0 and global_num_updates > 0:
            if num_updates + 1 > max_num_updates:
                # we log once per epoch

                # if we apply sequential test, then we should avoiding missing the results!
                gt_test_score = self.test(
                    test_config=test_config,
                    stage=stage,
                    eval_data_paths=eval_data_paths,
                    eval_label_paths=eval_label_paths,
                    is_2d=is_2d,
                    included_2d_pl=included_2d_pl,
                    included_2d_slices=included_2d_slices,
                    global_num_updates=global_num_updates,
                    is_save_for_paper=is_save_for_paper,
                    save_for_paper_path=save_for_paper_path,
                )
                # sets the model in training mode
                if is_2d:
                    raise NotImplementedError()
                else:
                    self.model.train()
                    self.model.testing = False
            
            # save model checkpoint
            if num_updates + 1 > max_num_updates and self.eccv_save_dir is not None:
                chpt_save_path = os.path.join(self.eccv_save_dir ,"model-{}-iter{}.pt".format(stage, global_num_updates))
                logger.info(f"""SAVE - {stage} - iter{global_num_updates}, num_updates: {num_updates}/{max_num_updates},\
                        num_epochs: {num_epochs}, to {chpt_save_path}""")
                torch.save(self.model.state_dict(), chpt_save_path)


            is_stop, is_stop_per_pl = self.should_stop(
                global_num_updates,
                num_updates,
                max_num_updates,                
                stage,
                is_2d=is_2d,
                included_2d_pl=included_2d_pl,
                included_2d_slices=included_2d_slices,
            )

            global_num_updates += 1
            num_updates += 1
            if is_stop:
                return is_stop, is_stop_per_pl, global_num_updates, num_updates

        # if not isinstance(self.scheduler, ReduceLROnPlateau):
        #     self.scheduler.step()
        #     logger.info(f"TRAIN - {stage} - iter{global_num_updates} - scheduler.last_epoch: {self.scheduler.last_epoch}, optm_lr: '{self.optimizer.param_groups[0]['lr']}'")
        return False, is_stop_per_pl, global_num_updates, num_updates

    def should_stop(
        self, 
        global_num_updates,
        num_updates, 
        max_num_updates, 
        stage,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
    ):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)

        return (is_stop_entirely, stop_per_pl in 2D cases)
        """
        if num_updates + 1 > max_num_updates:
            logger.info(f'TRAIN - {stage} - iter{global_num_updates}, Maximum number of updates {num_updates} exceeded {max_num_updates}.')
            return True, None

        min_lr = 1e-6
        if is_2d:
            raise NotImplementedError()            
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'TRAIN - {stage} - iter{global_num_updates}, Learning rate below the minimum {min_lr}.')
            return True, None

        return False, None

    def validate(self, is_2d=False, included_2d_pl=None, included_2d_slices=None):
        if is_2d:
            raise NotImplementedError()            
        else:
            val_losses = utils.RunningAverage()
            val_scores = utils.RunningAverage()

        if not len(self.val_loaders):
            return 0, 0
        with torch.no_grad():
            for i, t in enumerate(self.val_loaders):
                if is_2d:  
                    raise NotImplementedError()                    
                else:
                    input, target, _, weight, _ = self._split_training_batch(t)

                    output, loss, loss_seg = self._forward_pass(input, target, weight=weight, pos_weight=None, is_2d=is_2d, plane_key=None)
                    val_losses.update(loss_seg.item(), self._batch_size(input))

                    # if model contains final_activation layer for normalizing logits apply it, otherwise
                    # the evaluation metric will be incorrectly computed
                    if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                        output = self.model.final_activation(output)

                    eval_score = self.eval_criterion(output, target)
                    val_scores.update(eval_score.item(), self._batch_size(input))

                if self.max_validate_iters is not None and self.max_validate_iters <= i:
                    # stop validation
                    break
        if is_2d:
            raise NotImplementedError()            
        return val_losses.avg, val_scores.avg

    def test(
        self,
        test_config,
        stage,
        eval_data_paths,
        eval_label_paths,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        global_num_updates=0,
        is_save_for_paper=False,
        save_for_paper_path=None,
    ):
        """eval hold out test set during dynamic training"""
        count = 0
        all_metrics = []
        if len(eval_data_paths) < 1 or len(eval_label_paths) < 1:
            return
        if is_save_for_paper:
            assert os.path.exists(save_for_paper_path)
            test_info_for_paper = {}
        for new_data_path, new_label_path in zip(eval_data_paths, eval_label_paths):
            predictions = _gen_predictions(
                model=self.model,
                test_config=test_config,
                MODEL_PRED_DIR=None,
                new_data_path=new_data_path,
                output_pred_path=None,
                save_file=False,
                is_2d=is_2d,
                included_2d_pl=included_2d_pl,
                included_2d_slices=included_2d_slices,
            )
            logger.info(f"TEST - {stage} - iter{global_num_updates}, generated predictions for {count}/{len(eval_data_paths)}, {new_data_path}")
            metrics = _gen_evaluations(
                eval_criterion=self.eval_criterion,
                label_path=new_label_path,
                pred_path=predictions,
                propagate_pred_path=None,
                existing_label_slices=None,
                is_2d=is_2d,
            )
            logger.info(f"TEST - {stage} - iter{global_num_updates}, metrics = {metrics}, {count}/{len(eval_data_paths)}, {new_data_path}")
            all_metrics.append(metrics)

            if is_save_for_paper:
                assert os.path.exists(save_for_paper_path)
                test_info_for_paper[count] = {
                    "test_data_path": new_data_path,
                    "test_label_path": new_label_path,
                    "predictions": predictions,
                }
            count += 1
        if is_save_for_paper:
            pickle_file_path = os.path.join(save_for_paper_path, "{}-iter{}-test.pickle".format(stage, global_num_updates))
            with open(pickle_file_path, 'wb') as handle:
                pickle.dump(test_info_for_paper, handle)
            logger.info(f"saved test_info_for_paper to {pickle_file_path}")

        summary_metrics = {}
        for k, _ in metrics.items():
            score_list = [x[k] for x in all_metrics]
            summary_metrics[k] = {
                "number": len(score_list),
                "mean": np.round(np.mean(score_list), 2),
                "std": np.round(np.std(score_list), 2),
                "min": np.round(np.min(score_list), 2),
                "max": np.round(np.max(score_list), 2),
            }        
        logger.info(f"TEST_SUMMARY - {stage} - iter{global_num_updates}, summary_metrics: {summary_metrics}")
        
        return summary_metrics

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        assert len(t) == 5
        input, target, weighted_target, weight, sample_id_tensor = t
        return input, target, weighted_target, weight, sample_id_tensor

    def _get_kd_loss_param(self, output, output_prev, target):
        """
        get lambda and mask for kd loss
        output: the logits map from current model
        output_prev: the logits map from previous model
        target: ground truth map        
        """

        mask_kd = None
        if self.is_kd_mask:
            # option 1: class uncertainty from output_prev to weight the kd loss
            mask_kd = output_prev.detach().clone()
            mask_kd = mask_kd * mask_kd + (1.0 - mask_kd) * (1.0 - mask_kd)

        return self.lambda_kd, mask_kd

    def _get_con_loss_param(self, output, output_prev, target):
        """
        get lambda and mask for kd loss
        output: the logits map from current model
        output_prev: the logits map from previous model
        target: ground truth map        
        """        
        return self.lambda_con, self.lambda_con_early, self.lambda_conbr, self.lambda_con_late

    def _forward_pass(self, input, target, weight=None, pos_weight=None, is_2d=False, plane_key=None):
        """
        lambda_kd: the lambda on kd loss
        mask_kd: mask used for knowledge distillation loss        
        """        
        # forward pass
        if is_2d:
            raise NotImplementedError()
        else:
            # print("| DEBUG 2 - target.size() = {}".format(target.size()))
            output_target, output_kd, output_feature, output_feature_early, output_feature_conbr, output_feature_late = self.model(input)
            
        if is_2d:
            raise NotImplementedError()
        else:
            # segmentation loss
            loss_ = self.loss_criterion(output_target, target, weight, pos_weight)
        loss_seg_ = loss_.detach().clone()

        # model_prev's output for knowledge distillation
        if self.is_model_prev_ready and (
            (self.loss_criterion_kd is not None)
            or (self.loss_criterion_con is not None)
            or (self.loss_criterion_con_early is not None)
            or (self.loss_criterion_conbr is not None)
            or (self.loss_criterion_con_late is not None)
        ):
            output_prev_target, _, output_prev_feature, output_prev_feature_early, output_prev_feature_conbr, output_prev_feature_late = self.model_prev(input)
            lambda_kd, mask_kd = self._get_kd_loss_param(output_target, output_prev_target, target)
            lambda_con, lambda_con_early, lambda_conbr, lambda_con_late = self._get_con_loss_param(output_target, output_prev_target, target)
            logger.info(f"kd: lambda_kd={lambda_kd}, is_lwf={self.is_lwf}, is_kd_mask={self.is_kd_mask}")
            logger.info(f"con: lambda_con={lambda_con}, is_edge_only_con={self.is_edge_only_con}")
            logger.info(f"con_early: lambda_con_early={lambda_con_early}, is_edge_only_con_early={self.is_edge_only_con_early}")
            logger.info(f"conbr: lambda_conbr={lambda_conbr}, is_edge_only_conbr={self.is_edge_only_conbr}")
            logger.info(f"con_late: lambda_con_late={lambda_con_late}, is_edge_only_con_late={self.is_edge_only_con_late}")

            if self.loss_criterion_kd is not None and lambda_kd > 0:                            
                # knowedge distillation loss, same signature as loss_criterion, output_prev serve as target, mask_kd serve as (spatial) weight
                # we use the kd head to predict previous model's target head
                # whether we reproduce the Learning without Forgetting benchmark, in binary case, we should use single head for LWF            
                if self.is_lwf:
                    loss_kd_ = self.loss_criterion_kd(output_target, output_prev_target, mask_kd, pos_weight)
                else:
                    loss_kd_ = self.loss_criterion_kd(output_kd, output_prev_target, mask_kd, pos_weight)
                loss_ += lambda_kd * loss_kd_

            if self.loss_criterion_con is not None and lambda_con > 0:
                anchor_features, contrast_feature, contrib_mask_pos, contrib_mask_neg, conf_mask = pre_contractive_pixel(
                    output_feature, 
                    target, 
                    output_prev_feature, 
                    output_prev_target,
                    is_edge_only=self.is_edge_only_con,
                )
                loss_con_ = self.loss_criterion_con(anchor_features, contrast_feature, contrib_mask_pos, contrib_mask_neg, conf_mask)
                if loss_con_ is not None:
                    loss_ += lambda_con * loss_con_
            if self.loss_criterion_con_early is not None and lambda_con_early > 0:
                anchor_features_early, contrast_feature_early, contrib_mask_pos_early, contrib_mask_neg_early, conf_mask_early = pre_contractive_pixel(
                    output_feature_early, 
                    target, 
                    output_prev_feature_early, 
                    output_prev_target,
                    is_edge_only=self.is_edge_only_con_early,
                )                                    
                loss_con_early_ = self.loss_criterion_con_early(anchor_features_early, contrast_feature_early, 
                    contrib_mask_pos_early, contrib_mask_neg_early, conf_mask_early)
                if loss_con_early_ is not None:
                    loss_ += lambda_con_early * loss_con_early_
            if self.loss_criterion_conbr is not None and lambda_conbr > 0:
                anchor_features_conbr, contrast_feature_conbr, contrib_mask_pos_conbr, contrib_mask_neg_conbr, conf_mask_conbr = pre_contractive_pixel(
                    output_feature_conbr, 
                    target, 
                    output_prev_feature_conbr, 
                    output_prev_target,
                    is_edge_only=self.is_edge_only_conbr,
                )                                    
                loss_conbr_ = self.loss_criterion_conbr(anchor_features_conbr, contrast_feature_conbr, 
                    contrib_mask_pos_conbr, contrib_mask_neg_conbr, conf_mask_conbr)
                if loss_conbr_ is not None:
                    loss_ += lambda_conbr * loss_conbr_
            if self.loss_criterion_con_late is not None and lambda_con_late > 0:
                anchor_features_late, contrast_feature_late, contrib_mask_pos_late, contrib_mask_neg_late, conf_mask_late = pre_contractive_pixel(
                    output_feature_late, 
                    target, 
                    output_prev_feature_late, 
                    output_prev_target,
                    is_edge_only=self.is_edge_only_con_late,
                )                                    
                loss_con_late_ = self.loss_criterion_con_late(anchor_features_late, contrast_feature_late, 
                    contrib_mask_pos_late, contrib_mask_neg_late, conf_mask_late)
                if loss_con_late_ is not None:
                    loss_ += lambda_con_late * loss_con_late_

        return output_target, loss_, loss_seg_

    def _is_best_eval_score(self, eval_score, plane_key=None):
        if plane_key is not None:
            if self.eval_score_higher_is_better:
                is_best = eval_score > self.best_eval_score[plane_key]
            else:
                is_best = eval_score < self.best_eval_score[plane_key]

            if is_best:
                self.best_eval_score[plane_key] = eval_score
        else:
            if self.eval_score_higher_is_better:
                is_best = eval_score > self.best_eval_score
            else:
                is_best = eval_score < self.best_eval_score

            if is_best:
                self.best_eval_score = eval_score

        return is_best

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
