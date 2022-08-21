# eccv meta experiment on nci

device=4
# set eccv params below
is_conbr_head=1
is_kd_head=1

is_lwf=0
is_kd_loss=1
is_kd_mask=1
lambda_kd=0.4
temperature_kd=1.0

is_con_loss=1
lambda_con=0.1
temperature_con=1.0
# this should always be ZERO!
is_edge_only_con=0

is_con_early_loss=1
lambda_con_early=0.1
temperature_con_early=1.0
is_edge_only_con_early=1

is_conbr_loss=1
lambda_conbr=0.1
temperature_conbr=1.0
# this should always be ZERO!
is_edge_only_conbr=0

is_con_late_loss=1
lambda_con_late=0.1
temperature_con_late=1.0
is_edge_only_con_late=1


# set generate parameter below
online_legacy_capacity=2
online_max_num_iterations=150
random_seed=42
random_seed_shuffle_train_set=1
init_train_set="none"
online_train_set="train12"
test_set="test"
n_train1=0
n_train2=64
n_test=16
n_offline_first=-1
n_online_first=-1
proxy_label_weights_algo="none"

# experiment folder name
exp_folder="ft-rp${online_legacy_capacity}-nit${online_max_num_iterations}"
if [ $is_conbr_head == 1 ]
then
  echo "is_conbr_head"
  exp_folder="${exp_folder}-conH"
fi
if [ $is_kd_head == 1 ]
then
  echo "is_kd_head"
  exp_folder="${exp_folder}-kdH"
fi
if [ $is_kd_loss == 1 ]  
then
  echo "is_kd_loss"
  exp_folder="${exp_folder}-kdL-lwf${is_lwf}-kdM${is_kd_mask}-kdX${lambda_kd}-kdT${temperature_kd}"
fi
if [ $is_con_loss == 1 ]
then
  echo "is_con_loss"
  exp_folder="${exp_folder}-conL-conX${lambda_con}-conT${temperature_con}-conE${is_edge_only_con}"
fi
if [ $is_con_early_loss == 1 ]
then
  echo "is_con_early_loss"
  exp_folder="${exp_folder}-coneL-coneX${lambda_con_early}-coneT${temperature_con_early}-coneE${is_edge_only_con_early}"
fi
if [ $is_conbr_loss == 1 ]
then
  echo "is_conbr_loss"
  exp_folder="${exp_folder}-conbrL-conbrX${lambda_conbr}-coneT${temperature_conbr}-coneE${is_edge_only_conbr}"
fi
if [ $is_con_late_loss == 1 ]
then
  echo "is_con_late_loss"
  exp_folder="${exp_folder}-conlL-conlX${lambda_con_late}-conlT${temperature_con_late}-conlE${is_edge_only_con_late}"
fi

# log file name
log_name="train-rs${random_seed}-rsTR${random_seed_shuffle_train_set}"

# mahler dev8
DATA_ROOT="/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/all"
EXP_ROOT="/data/<USERNAME>/checkpoints/eccv/eccv_nci_v2/${exp_folder}"
CONFIG_ROOT="/home/<USERNAME>/Projects/pytorch-3dunet-dev/scripts/eccv_nci/config_eccv"

mkdir -p "${EXP_ROOT}"
mkdir -p "${EXP_ROOT}/${log_name}"
cp "/home/<USERNAME>/Projects/pytorch-3dunet-dev/scripts/eccv_nci/eccv_nci.sh" "${EXP_ROOT}/${log_name}/"
echo "EXP_ROOT=${EXP_ROOT}"
echo "log_name=${EXP_ROOT}/${log_name}"

DATA_SUFFIX="_image_norm_crop_resize.nrrd"
LABEL_SUFFIX="_label_binary_crop_resize.nrrd"

# generate log file name
CUDA_VISIBLE_DEVICES=$device nohup python -u pytorch3dunet/train_dynamic_v2.py \
--data_root $DATA_ROOT \
--exp_root $EXP_ROOT \
--log_name "${log_name}.log" \
--config_root $CONFIG_ROOT \
--random_seed $random_seed \
--random_seed_shuffle_train_set $random_seed_shuffle_train_set \
--n_train1 $n_train1 \
--n_train2 $n_train2 \
--n_offline_first $n_offline_first \
--n_online_first $n_online_first \
--n_test $n_test \
--init_train_set $init_train_set \
--online_train_set $online_train_set \
--test_set $test_set \
--data_path_suffix $DATA_SUFFIX \
--label_path_suffix $LABEL_SUFFIX \
--train_val_p 1.0 \
--online_legacy_capacity $online_legacy_capacity \
--online_legacy_capacity_from_offline 0 \
--online_new_capacity 1 \
--online_new_mcmc_capacity 0 \
--init_max_num_iterations 0 \
--online_max_num_iterations $online_max_num_iterations \
--online_annotation_type "full_3d" \
--online_annotation_rounds 1 \
--online_annotation_actions_per_round 1 \
--sample_selection_weights_algo "none" \
--loss_weights_algo "none" \
--proxy_label_weights_algo "${proxy_label_weights_algo}" \
--propose_online_annotation_algo "none" \
--slice_confidence_neib_radius 0 \
--proxy_label_gen_algo "none" \
--is_conbr_head $is_conbr_head \
--is_kd_head $is_kd_head \
--is_kd_loss $is_kd_loss \
--is_lwf $is_lwf \
--is_kd_mask $is_kd_mask \
--lambda_kd $lambda_kd \
--temperature_kd $temperature_kd \
--is_con_loss $is_con_loss \
--lambda_con $lambda_con \
--temperature_con $temperature_con \
--is_edge_only_con $is_edge_only_con \
--is_con_early_loss $is_con_early_loss \
--lambda_con_early $lambda_con_early \
--temperature_con_early $temperature_con_early \
--is_edge_only_con_early $is_edge_only_con_early \
--is_conbr_loss $is_conbr_loss \
--lambda_conbr $lambda_conbr \
--temperature_conbr $temperature_conbr \
--is_edge_only_conbr $is_edge_only_conbr \
--is_con_late_loss $is_con_late_loss \
--lambda_con_late $lambda_con_late \
--temperature_con_late $temperature_con_late \
--is_edge_only_con_late $is_edge_only_con_late \
--is_save_for_eccv \
&>"${EXP_ROOT}/nohup-${log_name}.out"&




