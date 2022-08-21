""" dynamic training for brats2015 dataset """

from pytorch3dunet.unet3d.config import load_config_dynmaic
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.trainer import MultiStageTrainer
import torch

# logger = get_logger('DynamicTrainingSetup')

def main():
    # Load and log experiment configuration
    args = load_config_dynmaic()

    logger = get_logger("DynamicTrainingSetup", filename=args.log_dir)    
    logger.info("===namespace===")
    logger.info(args)

    logger.info(f'Seed the RNG for all devices with {args.random_seed}')
    torch.manual_seed(args.random_seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.online_dice_threshold > 0 and args.online_round_termination_criterion in ["max_model_human", "human_only", "model_only"]:
        raise NotImplementedError()
    else:
        # create trainer
        print(f"MultiStageTrainer")
        multistage_trainer = MultiStageTrainer(
            data_root=args.data_root, 
            exp_root=args.exp_root, 
            init_train_config=args.init_train_config, 
            online_train_config=args.online_train_config, 
            test_config=args.test_config, 
            is_2d=args.is_2d, 
            random_seed=args.random_seed,
            random_seed_shuffle_train_set=args.random_seed_shuffle_train_set,
            random_seed_propose_annotation=args.random_seed_propose_annotation,
            n_train1=args.n_train1,
            n_train2=args.n_train2,
            n_offline_first=args.n_offline_first,
            n_online_first=args.n_online_first,
            n_test=args.n_test,
            sequential_evaluation_gaps=args.sequential_evaluation_gaps,
            init_train_set=args.init_train_set, 
            online_train_set=args.online_train_set,
            test_set=args.test_set,
            data_path_suffix=args.data_path_suffix,
            label_path_suffix=args.label_path_suffix,
            train_val_p=args.train_val_p,
            online_legacy_capacity=args.online_legacy_capacity,
            online_legacy_capacity_from_offline=args.online_legacy_capacity_from_offline,
            online_new_capacity=args.online_new_capacity,
            online_new_mcmc_capacity=args.online_new_mcmc_capacity,
            init_max_num_iterations=args.init_max_num_iterations,
            online_max_num_iterations=args.online_max_num_iterations,
            online_annotation_type=args.online_annotation_type,
            online_annotation_rounds=args.online_annotation_rounds,
            online_annotation_actions_per_round=args.online_annotation_actions_per_round,
            sample_selection_weights_algo=args.sample_selection_weights_algo,
            loss_weights_algo=args.loss_weights_algo,
            proxy_label_weights_algo=args.proxy_label_weights_algo,
            propose_online_annotation_algo=args.propose_online_annotation_algo,
            slice_confidence_neib_radius=args.slice_confidence_neib_radius,
            proxy_label_gen_algo=args.proxy_label_gen_algo,
            is_ueven_online_iter=args.is_ueven_online_iter,
            is_cold_start_interactive_seg=args.is_cold_start_interactive_seg,
            gen_label_adhoc_only=args.gen_label_adhoc_only,
            model_pred_conf_seed_thres=args.model_pred_conf_seed_thres,
            use_fixed_warmup_to_guide=args.use_fixed_warmup_to_guide,
            max_cpu_cores=args.max_cpu_cores,
            skip_train_first_annotation_rounds=args.skip_train_first_annotation_rounds,
            is_save_for_paper=args.is_save_for_paper,
            is_conbr_head=args.is_conbr_head == 1,
            is_kd_head=args.is_kd_head == 1,
            is_kd_loss=args.is_kd_loss == 1,
            is_lwf=args.is_lwf == 1,
            is_kd_mask=args.is_kd_mask == 1,
            lambda_kd=args.lambda_kd,
            temperature_kd=args.temperature_kd,
            is_con_loss=args.is_con_loss == 1, 
            lambda_con=args.lambda_con,
            temperature_con=args.temperature_con,
            is_edge_only_con=args.is_edge_only_con == 1,
            is_con_early_loss=args.is_con_early_loss == 1,
            lambda_con_early=args.lambda_con_early,
            temperature_con_early=args.temperature_con_early,
            is_edge_only_con_early=args.is_edge_only_con_early == 1,
            is_conbr_loss=args.is_conbr_loss == 1,
            lambda_conbr=args.lambda_conbr,
            temperature_conbr=args.temperature_conbr, 
            is_edge_only_conbr=args.is_edge_only_conbr == 1,
            is_con_late_loss=args.is_con_late_loss == 1,
            lambda_con_late=args.lambda_con_late,
            temperature_con_late=args.temperature_con_late,
            is_edge_only_con_late=args.is_edge_only_con_late == 1,
            eccv_save_dir=args.eccv_save_dir,
        )
        # Start dynamic training
        multistage_trainer.fit_dynamic()

if __name__ == '__main__':
    main()
