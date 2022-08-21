import glob
import sys
import os
from itertools import chain
# from multiprocessing import Lock

import numpy as np
import SimpleITK as sitk

import pytorch3dunet.augment.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.slice_builders import SliceBuilder
import collections

import torch


logger = get_logger('StandardDataset')
# lock = Lock()

def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))

def _itk_read_transform(tfm_path):
    return sitk.ReadTransform(tfm_path)

def _itk_read_inverse_transform(tfm_path):
    return sitk.ReadTransform(tfm_path).GetInverse()

def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch, list):
        if None in batch:
            batch = [x for x in batch if x is not None]
    if len(batch) < 1:
        return None
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def multiple_iter_datasets_collate(batch_dict_list):
    """the batch now should be a dictionary"""
    keys = list(batch_dict_list[0].keys())
    res = {}
    for k in keys:
        res[k] = default_prediction_collate([x[k] for x in batch_dict_list])
    return res

class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)

class MultipleIterDataset(Dataset):
    """iterate multiple datasets at the same time and return items in a dictionary"""
    def __init__(self, datasets, len_option="min", is_test=False) -> None:
        """datasets: dict of datasets
        len_option: "min", "max", "mean"
        """
        super(MultipleIterDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        for k, d in datasets.items():
            assert len(d) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = datasets
        self.lens = {k: len(d) for k, d in datasets.items()}
        self.lens_min = np.min([v for _, v in self.lens.items()])
        self.lens_max = np.max([v for _, v in self.lens.items()])
        self.lens_mean = np.mean([v for _, v in self.lens.items()])
        self.len_option = len_option
        self.is_test = is_test
        if self.is_test:
            assert self.len_option == "max"
        if self.len_option == "min":
            self._len = self.lens_min
        elif self.len_option == "max":
            self._len = self.lens_max
        elif self.len_option == "mean":
            self._len = self.lens_mean
        else:
            raise NotImplementedError()
        
        # import pdb; pdb.set_trace()
        # print("debug")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        res = {}
        for k, d in self.datasets.items():
            if self.is_test:
                if idx > len(d) - 1:
                    res[k] = None
                else:
                    res[k] = d[idx]
            else:
                if idx > len(d) - 1:
                    res[k] = d[np.random.randint(len(d))]
                elif self._len < len(d):
                    res[k] = d[np.random.randint(len(d))]
                else:
                    res[k] = d[idx]
        return res


class PreloadContent:
    """preload dataset and slices"""

    def __init__(
        self,
        file_path,
        label_path,
        weighted_label_path,
        loss_weight_path,
        transform_path,
        original_label_path,
        original_file_path,
        slice_builder_config,
        mcmc_spatial_prior,
        transformer_config,
        phase,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,        
    ):
        """
            weighted_label_path: normally weighted combination of previous model's prediction and current label
        """

        assert phase in ['train', 'val', 'test']        

        self.phase = phase
        self.file_path = file_path
        # self.sample_id = int(self.file_path.split("/")[-1].split(".")[0].split("_")[1])
        self.sample_id = 0
        self.label_path = label_path
        self.weighted_label_path = weighted_label_path
        self.loss_weight_path = loss_weight_path
        self.transform_path = transform_path
        self.original_label_path = original_label_path
        self.original_file_path = original_file_path

        self.is_2d = is_2d
        self.included_2d_pl = included_2d_pl
        self.included_2d_slices = included_2d_slices

        # raw image volume
        assert self.file_path is not None
        self.raw = _itk_read_array_from_file(self.file_path)
        self.vol_shape = self.raw.shape

        if self.is_2d:
            if included_2d_pl is None:
                included_2d_pl = ["z", "y", "x"]
            if included_2d_slices is None:
                included_2d_slices = {
                    "z": list(range(self.vol_shape[0])),
                    "y": list(range(self.vol_shape[1])),
                    "x": list(range(self.vol_shape[2])),
                }
        # label volume: should always be the proxy label!
        if self.label_path is not None:
            if isinstance(self.label_path, str):
                self.label = _itk_read_array_from_file(self.label_path)
            elif isinstance(self.label_path, np.ndarray):
                self.label = self.label_path
            else:
                raise TypeError()
        else:
            self.label = None

        if self.label is not None:
            if self.label.max() != 1 or self.label.min() != 0:
                unq_val_ = set(np.unique(self.label).astype(np.int32).tolist())
                logger.info(f"strange label found at self.label_path, unique values {unq_val_}")
                self.label[self.label != 1] = 0.0

        # weighted label volume: should always be the weighted combination of model's prediction and current label
        if self.weighted_label_path is not None:
            if isinstance(self.weighted_label_path, str):
                self.weighted_label = _itk_read_array_from_file(self.weighted_label_path)
            elif isinstance(self.weighted_label_path, np.ndarray):
                self.weighted_label = self.weighted_label_path
            else:
                raise TypeError()
        else:
            self.weighted_label = None


        if self.loss_weight_path is not None:
            assert isinstance(self.loss_weight_path, np.ndarray)
            self.loss_weight = self.loss_weight_path
        else:
            self.loss_weight = np.ones(self.vol_shape).astype(np.float32)
        
        if self.transform_path is not None:
            self.itk_transform = _itk_read_transform(self.transform_path)
        else:
            self.itk_transforms = None

        if self.original_label_path is not None:
            self.original_label = _itk_read_array_from_file(self.original_label_path)
        else:
            self.original_label = None

        min_value, max_value, mean, std = self.ds_stats()

        self.transformer = transforms.get_transformer(
            transformer_config, 
            min_value=min_value, 
            max_value=max_value,
            mean=mean, 
            std=std
        )
        self.raw_transform = self.transformer.raw_transform()
        self.label_transform = self.transformer.label_transform()

        slice_builder = SliceBuilder(
            vol_shape=self.raw.shape,
            patch_shape=slice_builder_config["patch_shape"],
            stride_shape=slice_builder_config["stride_shape"],
            label=self.label,
            is_2d=is_2d,
            included_2d_pl=included_2d_pl,
            included_2d_slices=included_2d_slices,
            threshold=slice_builder_config["threshold"], 
            slack_acceptance=slice_builder_config["slack_acceptance"], 
            threshold_count=slice_builder_config["threshold_count"],
            max_sample_size=slice_builder_config["max_sample_size"],
            mcmc_spatial_prior=mcmc_spatial_prior,
            mcmc_chain_length=slice_builder_config["mcmc_chain_length"],
            mcmc_sample_size=slice_builder_config["mcmc_sample_size"],
        )

        self.raw_slices = slice_builder.raw_slices

        # import pdb; pdb.set_trace()
        # print("debug")
        if self.is_2d:
            self.patch_count = {}
            for k in self.raw_slices.keys():
                self.patch_count[k] = len(self.raw_slices[k])
                logger.info(f'Number of patches - {k}: {self.patch_count[k]}')
        else:
            self.patch_count = len(self.raw_slices)
            logger.info(f'Number of patches: {self.patch_count}')


    def ds_stats(self):
        # calculate global min, max, mean and std for normalization
        min_value, max_value, mean, std = calculate_stats([self.raw])
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

class StandardDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the nrrd files in hippocampus dataset, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(
        self, 
        preload_content,
        selected_2d_pl=None,
    ):
        self.preload_content = preload_content
        if self.preload_content.is_2d:
            self.raw_slices = self.preload_content.raw_slices[selected_2d_pl]
        else:
            self.raw_slices = self.preload_content.raw_slices
        self.selected_2d_pl = selected_2d_pl
    
    def __getitem__(self, idx):
        """
            in 2D cases, the returned tensor must be of shape [1, *, *], the first dimension must be one
            later on during testing stage,         
        """
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self._transform_patches(self.preload_content.raw, raw_idx, self.preload_content.raw_transform, self.preload_content.is_2d, self.selected_2d_pl)

        if self.preload_content.phase == 'test':            
            if self.preload_content.label is not None:
                label_patch_transformed = self._transform_patches(self.preload_content.label, raw_idx, self.preload_content.label_transform, self.preload_content.is_2d, self.selected_2d_pl)
                return raw_patch_transformed, raw_idx, label_patch_transformed
            # note that the third raw_idx is a place holder for none label
            return raw_patch_transformed, raw_idx, raw_idx
        else:
            label_patch_transformed = self._transform_patches(self.preload_content.label, raw_idx, self.preload_content.label_transform, self.preload_content.is_2d, self.selected_2d_pl)
            if label_patch_transformed.max() != 1 or label_patch_transformed.min() != 0:
                unq_val_ = set(np.unique(label_patch_transformed).astype(np.int32).tolist())
                logger.info(f"strange label found at label_patch_transformed, unique values {unq_val_}")
                label_patch_transformed[label_patch_transformed != 1] = 0.0

            weighted_label_patch_transformed = self._transform_patches(self.preload_content.weighted_label, raw_idx, self.preload_content.label_transform, self.preload_content.is_2d, self.selected_2d_pl)
            if self.preload_content.loss_weight is not None:
                loss_weight_patch_transformed = self._transform_patches(self.preload_content.loss_weight, raw_idx, self.preload_content.label_transform, self.preload_content.is_2d, self.selected_2d_pl)
                return raw_patch_transformed, label_patch_transformed, weighted_label_patch_transformed, loss_weight_patch_transformed, self.preload_content.sample_id
            return raw_patch_transformed, label_patch_transformed, weighted_label_patch_transformed, self.preload_content.sample_id

    @staticmethod
    def _transform_patches(dataset, label_idx, transformer, is_2d=False, selected_2d_pl=None):
        transformed_patches = []
        # get the label data and apply the label transformer
        dd = dataset[label_idx]
        if is_2d:
            if selected_2d_pl == "z":
                assert dd.shape[0] == 1 and dd.shape[1] > 1 and dd.shape[2] > 1
            elif selected_2d_pl == "y":
                assert dd.shape[0] > 1 and dd.shape[1] == 1 and dd.shape[2] > 1
                dd = np.transpose(dd, axes=(1, 0, 2))
            elif selected_2d_pl == "x":
                assert dd.shape[0] > 1 and dd.shape[1] > 1 and dd.shape[2] == 1
                dd = np.transpose(dd, axes=(2, 0, 1))
        transformed_patch = transformer(dd)
        return transformed_patch

    def __len__(self):
        if self.preload_content.is_2d:
            return self.preload_content.patch_count[self.selected_2d_pl]
        return self.preload_content.patch_count

    @classmethod
    def create_datasets(
        cls,
        dataset_config, 
        phase, 
        slice_builder_config_override = None,
        mcmc_spatial_prior = None,
        file_paths_override = None,
        label_paths_override = None,
        weighted_label_paths_override = None,
        loss_weight_paths_override = None,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        len_option="min",
        is_test=False,
    ):
        assert phase in ["train", "test"]
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # if phase == "test":
            # import pdb; pdb.set_trace()
        # load slice builder config
        if slice_builder_config_override is not None:
            slice_builder_config = slice_builder_config_override
        else:
            slice_builder_config = phase_config['slice_builder']
        # load files to process
        if file_paths_override is not None:
            file_paths = file_paths_override
        else:
            file_paths = phase_config['file_paths']
        if label_paths_override is not None:
            label_paths = label_paths_override
        else:
            label_paths = phase_config.get('label_paths', [None] * len(file_paths))

        if weighted_label_paths_override is not None:
            weighted_label_paths = weighted_label_paths_override
        else:
            weighted_label_paths = phase_config.get('weighted_label_paths', [None] * len(file_paths))

        if loss_weight_paths_override is not None:
            loss_weight_paths = loss_weight_paths_override
        else:
            loss_weight_paths = phase_config.get('loss_weight_paths', [None] * len(file_paths))

        transform_paths = phase_config.get('transform_paths', [None] * len(file_paths))
        original_label_paths = phase_config.get('original_label_paths', [None] * len(file_paths))
        original_file_paths = phase_config.get('original_file_paths', [None] * len(file_paths))

        datasets = []
        for file_path, label_path, weighted_label_path, loss_weight_path, transform_path, original_label_path, original_file_path in zip(
            file_paths, label_paths, weighted_label_paths, loss_weight_paths, transform_paths, original_label_paths, original_file_paths
        ):
            preload_content = PreloadContent(
                file_path,
                label_path,
                weighted_label_path,
                loss_weight_path,
                transform_path,
                original_label_path,
                original_file_path,
                slice_builder_config=slice_builder_config,
                mcmc_spatial_prior=mcmc_spatial_prior,
                transformer_config=transformer_config,
                phase=phase,
                is_2d=is_2d,
                included_2d_pl=included_2d_pl,
                included_2d_slices=included_2d_slices,        
            )
            logger.info(f'Loading {phase} set from: {file_path}...')
            if is_2d:
                dataset_dict = {}
                for k in included_2d_pl:
                    dataset_dict[k] =  cls(
                        preload_content,
                        selected_2d_pl=k,
                    )
                dataset = MultipleIterDataset(dataset_dict, len_option=len_option, is_test=is_test)
                # print(dataset[0])
                # import pdb; pdb.set_trace()
                # print("debug")
            else:
                dataset = cls(
                    preload_content,
                    selected_2d_pl=None,
                )
            datasets.append(dataset)
        return datasets

class IntermediateDataset(ConfigDataset):
    """intermediate dataset that takes stacked input, target and weight, normally sampled from buffer"""
    def __init__(self, input, target, weighted_target, weight, sample_id_tensor):
        self._sample_size = input.size(0)
        self.input = input
        self.target = target
        self.weighted_target = weighted_target
        self.weight = weight
        self.sample_id_tensor = sample_id_tensor

    def __getitem__(self, index):
        return self.input[index], self.target[index], self.weighted_target[index], self.weight[index], self.sample_id_tensor[index]
    
    def __len__(self):
        return self._sample_size

def get_intermediate_loader(
    intermediate_dataset,
    config,
):
    loaders_key = "loaders"
    assert loaders_key in config, 'Could not find data loaders configuration'
    loaders_config = config[loaders_key]
    num_workers = loaders_config.get('num_workers', 1)
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()
    return DataLoader(intermediate_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_train_val_loaders_dynamic(
    config, 
    legacy_dataset,
    legacy_dataset_from_offline,
    initial_data_paths, 
    initial_label_paths, 
    new_data_path,
    new_label_path,
    new_weighted_label_path,
    online_legacy_capacity = 10000,
    online_legacy_capacity_from_offline = 10000,
    online_new_capacity = 5000,
    online_new_mcmc_capacity = 5000,
    train_val_p = 0.8,
    random_seed = 42,
    is_2d=False,
    included_2d_pl=None,
    included_2d_slices=None,
    len_option="min",
    sample_selection_weights=None,
    loss_weights=None,
):
    loaders_key = "loaders"
    curr_dataset, train_dataset, val_dataset = _get_train_val_dataset_dynamic(
        config=config, 
        legacy_dataset=legacy_dataset, 
        legacy_dataset_from_offline=legacy_dataset_from_offline,
        initial_data_paths=initial_data_paths, 
        initial_label_paths=initial_label_paths, 
        new_data_path=new_data_path, 
        new_label_path=new_label_path, 
        new_weighted_label_path=new_weighted_label_path,
        online_legacy_capacity=online_legacy_capacity,
        online_legacy_capacity_from_offline=online_legacy_capacity_from_offline,
        online_new_capacity=online_new_capacity,
        online_new_mcmc_capacity=online_new_mcmc_capacity,
        train_val_p=train_val_p,
        random_seed=random_seed,
        is_2d=is_2d,
        included_2d_pl=included_2d_pl,
        included_2d_slices=included_2d_slices,
        len_option=len_option,
        sample_selection_weights=sample_selection_weights,
        loss_weights=loss_weights,
    )
    assert loaders_key in config, 'Could not find data loaders configuration'
    loaders_config = config[loaders_key]
    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    if is_2d:
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=multiple_iter_datasets_collate),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=multiple_iter_datasets_collate) if val_dataset is not None else []
        }, curr_dataset, train_dataset, val_dataset
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset is not None else []
    }, curr_dataset, train_dataset, val_dataset


def _get_train_val_dataset_dynamic(
    config, 
    legacy_dataset, 
    legacy_dataset_from_offline,
    initial_data_paths, 
    initial_label_paths, 
    new_data_path, 
    new_label_path,
    new_weighted_label_path,
    online_legacy_capacity = 10000,
    online_legacy_capacity_from_offline = 10000,
    online_new_capacity = 5000,
    online_new_mcmc_capacity = 5000,
    train_val_p = 0.8,
    random_seed = 42,
    is_2d=False,
    included_2d_pl=None,
    included_2d_slices=None,
    len_option="min",
    sample_selection_weights=None,
    loss_weights=None,
):
    loaders_key = "loaders"
    assert loaders_key in config, 'Could not find data loaders configuration'
    loaders_config = config[loaders_key]
     # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")

    if initial_data_paths is not None:
        assert legacy_dataset_from_offline is None and initial_label_paths is not None and len(initial_label_paths) == len(initial_data_paths)
        curr_dataset = ConcatDataset(
            StandardDataset.create_datasets(
                loaders_config, 
                phase='train',
                slice_builder_config_override = None,
                mcmc_spatial_prior = None,
                file_paths_override = initial_data_paths,
                label_paths_override = initial_label_paths,
                weighted_label_paths_override = initial_label_paths,
                loss_weight_paths_override = None,
                is_2d=is_2d,
                included_2d_pl=included_2d_pl,
                included_2d_slices=included_2d_slices,
                len_option=len_option,
                is_test=False,
            )
        )

        logger.info(f"INITIAL DATASET | total: {len(curr_dataset)} legacy: 0, new: 0, new_mcmc_keep: 0")
        n_train = int(len(curr_dataset) * train_val_p)
        n_val = len(curr_dataset) - n_train

        train_dataset, val_dataset = random_split(curr_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))
        return curr_dataset, train_dataset, val_dataset

    assert new_data_path is not None and new_label_path is not None and new_weighted_label_path is not None
    
    slice_builder_config_override = {
        "name": "SliceBuilder",
        "patch_shape": loaders_config["train"]["slice_builder"]["patch_shape"],
        "stride_shape": loaders_config["train"]["slice_builder"]["stride_shape"],
        "threshold": loaders_config["train"]["slice_builder"]["threshold"],
        "slack_acceptance": loaders_config["train"]["slice_builder"]["slack_acceptance"],
        "threshold_count": loaders_config["train"]["slice_builder"]["threshold_count"],
        "max_sample_size": online_new_capacity,
        "mcmc_chain_length": online_new_mcmc_capacity * 10,
        "mcmc_sample_size": online_new_mcmc_capacity,        
    }

    new_dataset = ConcatDataset(
        StandardDataset.create_datasets(
            loaders_config, 
            phase='train',
            slice_builder_config_override = slice_builder_config_override,
            mcmc_spatial_prior = sample_selection_weights,
            file_paths_override = [new_data_path],
            label_paths_override = [new_label_path],
            weighted_label_paths_override = [new_weighted_label_path],
            loss_weight_paths_override = [loss_weights],
            is_2d=is_2d,
            included_2d_pl=included_2d_pl,
            included_2d_slices=included_2d_slices,
            len_option=len_option,
            is_test=False,
        )
    )

    # build legacy datasets and concat datasets
    concate_datasets = []


    legacy_keep = None
    n_legacy = 0
    if legacy_dataset is not None:
        n_legacy = min(online_legacy_capacity, len(legacy_dataset))
        if n_legacy > 0:
            legacy_keep, _ = random_split(legacy_dataset, [n_legacy, len(legacy_dataset) - n_legacy], generator=torch.Generator().manual_seed(random_seed))
            concate_datasets.append(legacy_keep)

    legacy_keep_from_offline = None
    n_legacy_from_offline = 0
    if legacy_dataset_from_offline is not None:
        n_legacy_from_offline = min(online_legacy_capacity_from_offline, len(legacy_dataset_from_offline))
        if n_legacy_from_offline > 0:
            legacy_keep_from_offline, _ = random_split(
                legacy_dataset_from_offline, [n_legacy_from_offline, len(legacy_dataset_from_offline) - n_legacy_from_offline],
                generator=torch.Generator().manual_seed(random_seed)
            )
            concate_datasets.append(legacy_keep_from_offline)

    concate_datasets.append(new_dataset)
    curr_dataset = ConcatDataset(concate_datasets)
    logger.info(f"NEW DATASET | total: {len(curr_dataset)} legacy: {n_legacy}, lagecy_from_offline: {n_legacy_from_offline}, new: {len(new_dataset)}, new_mcmc_keep: 0")
    
    n_train = max(1, int(len(curr_dataset) * train_val_p))
    n_val = len(curr_dataset) - n_train
    if n_val > 0:
        train_dataset, val_dataset = random_split(curr_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))
    else:
        train_dataset, val_dataset = curr_dataset, None
    return curr_dataset, train_dataset, val_dataset

def get_test_loaders_dynamic(
    config,     
    new_data_path,
    new_label_path=None,
    is_2d=False,
    included_2d_pl=None,
    included_2d_slices=None,
    len_option="min",
):
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")

    test_datasets = StandardDataset.create_datasets(
        loaders_config, 
        phase='test',
        slice_builder_config_override = None,
        mcmc_spatial_prior = None,
        file_paths_override = [new_data_path],
        label_paths_override = [new_label_path] if new_label_path is not None else [None],
        loss_weight_paths_override = None,
        is_2d=is_2d,
        included_2d_pl=included_2d_pl,
        included_2d_slices=included_2d_slices,
        len_option=len_option,
        is_test=True,
    )

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # laoder_ = DataLoader(test_datasets[0], batch_size=batch_size, num_workers=num_workers, collate_fn=multiple_iter_datasets_collate)
    # import pdb; pdb.set_trace()
    
    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        if is_2d:
            collate_fn = multiple_iter_datasets_collate
        else:
            if hasattr(test_dataset, 'prediction_collate'):
                collate_fn = test_dataset.prediction_collate
            else:
                collate_fn = default_prediction_collate
        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=collate_fn)


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)

