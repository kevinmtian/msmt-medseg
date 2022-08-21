import argparse
from ast import parse
from errno import EFAULT
import typing
import os
import torch
import yaml
import json

from pytorch3dunet.unet3d import utils
from typing import Optional

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config

def _materialize_config(config_yaml):
    config = _load_config_yaml(config_yaml)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config

def load_config_dynmaic():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--exp_root", type=str, required=True)
    parser.add_argument("--config_root", type=str, required=True)
    parser.add_argument("--is_2d", default=False, action="store_true")
    parser.add_argument("--random_seed", type=int, default=42, required=True)
    parser.add_argument("--random_seed_shuffle_train_set", type=int, default=-1, required=False)
    parser.add_argument("--random_seed_propose_annotation", type=int, default=0, required=False)
    parser.add_argument("--n_train1", type=int, default=1, required=True)
    parser.add_argument("--n_train2", type=int, default=1, required=True)
    parser.add_argument("--n_offline_first", type=int, default=-1, required=False)
    parser.add_argument("--n_online_first", type=int, default=-1, required=False)
    parser.add_argument("--n_test", type=int, default=1, required=True)
    parser.add_argument("--sequential_evaluation_gaps", type=int, default=1, required=False)
    parser.add_argument("--init_train_set", type=str, default="train1", required=True)
    parser.add_argument("--online_train_set", type=str, default="train2", required=True)
    parser.add_argument("--test_set", type=str, default="test", required=True)
    parser.add_argument("--data_path_suffix", type=str, default="-reg.nrrd", required=True)
    parser.add_argument("--label_path_suffix", type=str, default="-whole-label-reg.nrrd", required=True)
    parser.add_argument("--train_val_p", type=float, default=0.8, required=True)
    parser.add_argument("--online_legacy_capacity", type=int, default=100, required=True)
    parser.add_argument("--online_legacy_capacity_from_offline", type=int, default=100, required=True)    
    parser.add_argument("--online_new_capacity", type=int, default=100, required=True)
    parser.add_argument("--online_new_mcmc_capacity", type=int, default=0, required=True)
    parser.add_argument("--init_max_num_iterations", type=int, default=100, required=True)
    parser.add_argument("--online_max_num_iterations", type=int, default=100, required=True)
    parser.add_argument("--online_annotation_type", type=str, default="full_3d", 
        choices=["full_3d", "slice-z", "slice-y", "slice-x", "slice-zy", "slice-zx", "slice-yx", "slice-zyx", "points"], required=True)
    parser.add_argument("--online_annotation_rounds", type=int, default=1, required=True, help="if 3d, then 1")
    parser.add_argument("--online_annotation_actions_per_round", type=int, default=1, required=True, help="if 3d, then 1")
    parser.add_argument("--sample_selection_weights_algo", type=str, default="residule-dev",
        choices=["none", "residule-dev", "residule-dice", "confidence-dist", "confidence-neib"], required=True)
    parser.add_argument("--loss_weights_algo", type=str, default="confidence-dist",
        # choices=["none", "residule-dev", "residule-dice", "confidence-dist", "confidence-neib", "confidence-slicereg_1.0"], 
        required=True)
    parser.add_argument("--proxy_label_weights_algo", type=str, default="confidence-dist",
        # choices=["none", "sigmoid-scaled", "inv-sigmoid-scaled", "sigmoid", "sigmoid-onlineonly", "residule-dev", "residule-dice", "confidence-dist", "confidence-neib"], 
        required=True)
    parser.add_argument("--propose_online_annotation_algo", type=str, default="random",
        choices=["none", "random", "residule-dev", "residule-dice", "confidence-dist", "largest_residule_cc", "confidence_cc",
            "largest_residule_cc_v2",
            "true_label_residule_random", "proxy_label_residule_random", "true_label_residule_worst_point", "proxy_label_residule_worst_point", "gt_largest_residule_cc_random"], required=True)
    parser.add_argument("--slice_confidence_neib_radius", type=int, default=3, required=True)
    parser.add_argument("--proxy_label_gen_algo", type=str, default="contributor-hard",
        choices=["none", "contributor-soft", "contributor-hard", "contributor-geo", "slice-only", "sanity_check", "fastgc_3d", "fastgc_contributor"], required=True)

    parser.add_argument("--is_ueven_online_iter", default=False, action="store_true")
    parser.add_argument("--is_cold_start_interactive_seg", default=False, action="store_true")
    parser.add_argument("--gen_label_adhoc_only", default=False, action="store_true")

    parser.add_argument("--buffer_size", default=16, type=int, help="buffer size, number of patches")
    parser.add_argument("--random_subsample_slots_each_batch", default=16, type=int, help="during ocl training, max number of candidate patches from the buffer")
    parser.add_argument("--retrieve_slots_each_batch", default=8, type=int, help="during ocl training, max number of patches retrieved from the buffer")

    parser.add_argument("--model_pred_conf_seed_thres", default=0.0, type=float, required=False)
    parser.add_argument("--use_fixed_warmup_to_guide", default=False, action="store_true")
    parser.add_argument("--max_cpu_cores", type=int, default=8, required=False)

    parser.add_argument("--online_dice_threshold", type=float, default=0.0, required=False)
    parser.add_argument("--online_round_termination_criterion", type=str, choices=["none", "max_model_human", "human_only", "model_only"], default="none", required=False)

    parser.add_argument("--skip_train_first_annotation_rounds", type=int, default=-1, required=False)
    parser.add_argument("--is_save_for_paper", default=False, action="store_true")

    parser.add_argument("--is_conbr_head", type=int, default=1, choices=[1, 0])
    parser.add_argument("--is_kd_head", type=int, default=1, choices=[1, 0])
    parser.add_argument("--is_kd_loss", type=int, default=0, choices=[1, 0])
    parser.add_argument("--is_lwf", type=int, default=0, choices=[1, 0])
    parser.add_argument("--is_kd_mask", type=int, default=0, choices=[1, 0])
    parser.add_argument("--lambda_kd", default=1.0, type=float)
    parser.add_argument("--temperature_kd", default=1.0, type=float)
    parser.add_argument("--is_con_loss", type=int, default=0, choices=[1, 0])
    parser.add_argument("--lambda_con", default=1.0, type=float)
    parser.add_argument("--temperature_con", default=1.0, type=float)
    parser.add_argument("--is_edge_only_con", type=int, default=0, choices=[1, 0])
    parser.add_argument("--is_con_early_loss", type=int, default=0, choices=[1, 0])
    parser.add_argument("--lambda_con_early", default=1.0, type=float)
    parser.add_argument("--temperature_con_early", default=1.0, type=float)
    parser.add_argument("--is_edge_only_con_early", type=int, default=0, choices=[1, 0])
    parser.add_argument("--is_conbr_loss", type=int, default=0, choices=[1, 0])
    parser.add_argument("--lambda_conbr", default=1.0, type=float)
    parser.add_argument("--temperature_conbr", default=1.0, type=float)
    parser.add_argument("--is_edge_only_conbr", type=int, default=0, choices=[1, 0])
    parser.add_argument("--is_con_late_loss", type=int, default=0, choices=[1, 0])
    parser.add_argument("--lambda_con_late", default=1.0, type=float)
    parser.add_argument("--temperature_con_late", default=1.0, type=float)
    parser.add_argument("--is_edge_only_con_late", type=int, default=0, choices=[1, 0])

    parser.add_argument("--log_name", type=str, default="train.log")
    parser.add_argument("--is_save_for_eccv", default=False, action="store_true")

    parser.add_argument("--model_checkpoint", type=str, default="none")
                
    args = parser.parse_args()
    args.log_dir = os.path.join(args.exp_root, args.log_name)
    # dir for save eccv models
    args.eccv_save_dir = None
    if args.is_save_for_eccv:
        args.eccv_save_dir = os.path.join(args.exp_root, args.log_name.rstrip(".log"))
    if (args.eccv_save_dir is not None) and (not os.path.exists(args.eccv_save_dir)):
        os.makedirs(args.eccv_save_dir)

    if os.path.exists(args.model_checkpoint):
        model_name = args.model_checkpoint.split("/")[-1]
        args.eccv_infer_save_dir = os.path.join(args.exp_root, model_name + "-infer")
        if not os.path.exists(args.eccv_infer_save_dir):
            os.makedirs(args.eccv_infer_save_dir)

    if False:
        # import pdb; pdb.set_trace()
        if args.gen_label_adhoc_only:
            args.log_dir = os.path.join(args.exp_root, "train-i{}-o{}-t{}-sspt{}-dt{}-{}-{}-sbst{}-{}-sshuf{}-spp{}-labeladhoconly.log".format(
                args.init_train_set,
                args.online_train_set,
                args.test_set,
                args.random_seed,
                args.n_train1,
                args.n_train2,
                args.n_test,
                args.n_offline_first,
                args.n_online_first,
                args.random_seed_shuffle_train_set,
                args.random_seed_propose_annotation,
            ))
        else:
            args.log_dir = os.path.join(args.exp_root, "train-i{}-o{}-t{}-sspt{}-dt{}-{}-{}-sbst{}-{}-sshuf{}-spp{}.log".format(
                args.init_train_set,
                args.online_train_set,
                args.test_set,
                args.random_seed,
                args.n_train1,
                args.n_train2,
                args.n_test,
                args.n_offline_first,
                args.n_online_first,
                args.random_seed_shuffle_train_set,
                args.random_seed_propose_annotation,
            ))
        if args.online_dice_threshold > 0 and args.online_round_termination_criterion != "none":
            args.log_dir = args.log_dir.split(".log")[0] + "-ods{}-otc{}.log".format(args.online_dice_threshold, args.online_round_termination_criterion)
            if args.is_ueven_online_iter:
                args.log_dir = args.log_dir.split(".log")[0] + "-uev.log"

    args.init_train_config = _materialize_config(os.path.join(args.config_root, "init_train_config.yaml"))
    args.online_train_config = _materialize_config(os.path.join(args.config_root, "online_train_config.yaml"))
    args.test_config = _materialize_config(os.path.join(args.config_root, "test_config.yaml"))
    return args

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
