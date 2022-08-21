import os

import h5py
from matplotlib.pyplot import axes
import numpy as np
import torch
from skimage import measure
import time

from pytorch3dunet.datasets.standard_dataset import (
    StandardDataset,
    MultipleIterDataset,
)
from pytorch3dunet.datasets.slice_builders import SliceBuilder
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.utils import remove_halo
import pytorch3dunet.unet3d.utils as utils
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
import SimpleITK as sitk

from tqdm import tqdm
from time import sleep

logger = get_logger('UNetPredictor')

def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))

def _itk_read_transform(tfm_path):
    return sitk.ReadTransform(tfm_path)

def _itk_read_inverse_transform(tfm_path):
    return sitk.ReadTransform(tfm_path).GetInverse()

def _itk_write_image_to_file(itk_image, output_path):
    sitk.WriteImage(itk_image, output_path, True)

def _itk_get_image_from_array(ref_image, arr):
    xxai = sitk.GetImageFromArray(arr)
    xxai.SetSpacing(ref_image.GetSpacing())
    xxai.SetOrigin(ref_image.GetOrigin())
    xxai.SetDirection(ref_image.GetDirection())    
    return xxai


def _itk_apply_transform(ref_image, input_image, itk_transform, interpolator=sitk.sitkLinear):
    """
    args:
        ref_image: itk image as ref image in the resampler
        input_image: itk image to apply the transform on
        itk_transform: itk transform to apply
        interpolator: sitkNearestNeighbor for label, sitk.sitkLinear for raw image
    return:
        output itk image from applying itk transform on input image    
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(itk_transform)
    output_image = resampler.Execute(input_image)
    return output_image


def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + suffix + '.h5')
    return output_file


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


class _AbstractPredictor:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, test_loader):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        # Create loss criterion
        self.loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        self.eval_criterion = get_evaluation_metric(config)
        self.device = self.config['device']

    def _batch_size(self, input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
            
    def _forward_pass(self, input, target, weight=None):
        # forward pass

        # print("| DEBUG 2 - target.size() = {}".format(target.size()))
        output, _, _, _, _, _ = self.model(input)

        # compute the loss
        if weight is None:
            # print("| DEBUG 3 - target.size() = {}".format(target.size()))
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _split_testing_batch(self, t):
        target = None
        if t is None:
            return None, None, None
        if len(t) == 2:
            input, idx = t
            input = input.to(self.device)
        else:
            input, idx, target = t
            input = input.to(self.device)
            if not isinstance(target, tuple):
                target = target.to(self.device)
        return input, idx, target

    def _test_batchwise(self, device, test_loader_list):

        raise NotImplementedError()
        logger.info('Testing...')

        test_losses = utils.RunningAverage()
        test_scores = utils.RunningAverage()
        
        with torch.no_grad():
            for test_loader in test_loader_list:
                logger.info(f"Evaluating '{test_loader.dataset.file_path}'...{len(test_loader)} patches")
                for _, t in enumerate(test_loader):

                    # logger.info(f'Validation iteration {i}')

                    input, _, target = self._split_testing_batch(device, t)

                    output, loss = self._forward_pass(input, target, None)

                    # import pdb; pdb.set_trace()

                    test_losses.update(loss.item(), self._batch_size(input))

                    # if model contains final_activation layer for normalizing logits apply it, otherwise
                    # the evaluation metric will be incorrectly computed
                    if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                        output = self.model.final_activation(output)

                    # if i % 100 == 0:
                    #     self._log_images(input, target, output, 'val_')

                    
                    eval_score = self.eval_criterion(output, target)
                    # import pdb; pdb.set_trace()
                    test_scores.update(eval_score.item(), self._batch_size(input))

                    # if self.sample_plotter is not None:
                    #     self.sample_plotter(i, input, output, target, 'val')

                    # if self.max_validate_iters is not None and self.max_validate_iters <= i:
                    #     # stop validation
                    #     break

            # self._log_stats('val', test_losses.avg, test_scores.avg)
            logger.info(f'Testing finished. Loss: {test_losses.avg}. Evaluation score: {test_scores.avg}')
            return test_scores.avg


    def evaluate_inv_transformed(self, test_loader_list, suffix="_predictions"):
        """evaluate the inverse transformed predictions vs. original labels"""

        # read predictions
        test_img_scores_inv = utils.RunningAverage()

        for test_loader in test_loader_list:
            logger.info(f"Processing '{test_loader.dataset.file_path}'...")
            output_file = _get_output_file(dataset=test_loader.dataset, suffix=suffix, output_dir=self.output_dir)
            with h5py.File(output_file, 'r') as f:
                predictions = f['predictions'][...][:, :, :]

            original_raw_image = _itk_read_image_from_file(test_loader.dataset.original_file_path)
            raw_image = _itk_read_image_from_file(test_loader.dataset.file_path)
            pred_image = _itk_get_image_from_array(raw_image, predictions)

            inv_itk_transform = test_loader.dataset.itk_transforms[0].GetInverse()

            # inv transform predictions
            original_pred_image = _itk_apply_transform(original_raw_image, pred_image, inv_itk_transform, interpolator=sitk.sitkLinear)
            original_predictions = sitk.GetArrayFromImage(original_pred_image)
            original_label = test_loader.dataset.original_labels[0]

            # compute dice
            eval_metric_inv = self.eval_criterion(
                torch.tensor(original_predictions).squeeze(), 
                torch.tensor(original_label).squeeze()
            )             
            logger.info(f'eval_metric_inv = {eval_metric_inv}')
            test_img_scores_inv.update(eval_metric_inv.item(), 1)

            # save original_pred_image as nrrd
            _, h5_file_name = os.path.split(output_file)
            pred_file = os.path.join(self.output_dir, os.path.splitext(h5_file_name)[0] + "inv_pred" + '.nrrd')
            if not os.path.exists(pred_file):
                _itk_write_image_to_file(original_pred_image, pred_file)
                logger.info(f"Save inv prediction image to {pred_file}")

        logger.info(f'AVG test_img_scores_inv: {test_img_scores_inv.avg}')

    def gen_predictions(
        self,
        test_loader_list,
        output_pred_path, 
        save_file=True,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        fuse_2d_option="mean",
    ):
        """only generate predictions assuming no labels provided, no evaluation done
            output_pred_path: pred path h5 file

        if is_2d, then return by pl prediction and predictions 2d fused        
        """
        
        
        output_file = output_pred_path
        if save_file and output_file:
            # create destination H5 file
            h5_output_file = h5py.File(output_file, 'w')
        else:
            h5_output_file = None

        if is_2d:
            for k in included_2d_pl:
                # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
                self.model[k].eval()
                # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
                self.model[k].testing = True
        else:
            # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
            self.model.eval()
            # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
            self.model.testing = True
        assert len(test_loader_list) == 1
        for test_loader in test_loader_list:
            
            if is_2d:
                k_0 = included_2d_pl[0]
                assert isinstance(test_loader.dataset, MultipleIterDataset) and isinstance(test_loader.dataset.datasets[k_0], StandardDataset)
                logger.info(f"Processing '{test_loader.dataset.datasets[k_0].preload_content.file_path}'...")
                volume_shape = self.volume_shape(test_loader.dataset.datasets[k_0].preload_content)
                prediction_map = {k: np.zeros(volume_shape, dtype='float32') for k in included_2d_pl}
                # label_map = {k: np.zeros(volume_shape, dtype='float32') for k in included_2d_pl}
                normalization_mask = {k: np.zeros(volume_shape, dtype='float32') for k in included_2d_pl}
                for k in included_2d_pl:
                    logger.info(f'The shape of the output prediction maps (CDHW) - {k}: {volume_shape}')
                    logger.info(f'Running prediction on {k} - {len(test_loader)} batches...')
            else:
                assert isinstance(test_loader.dataset, StandardDataset)
                logger.info(f"Processing '{test_loader.dataset.preload_content.file_path}'...")
                # dimensionality of the the output predictions
                volume_shape = self.volume_shape(test_loader.dataset.preload_content)
                prediction_map = np.zeros(volume_shape, dtype='float32')
                # label_map = np.zeros(volume_shape, dtype='float32')
                normalization_mask = np.zeros(volume_shape, dtype='float32')
                logger.info(f'The shape of the output prediction maps (CDHW): {volume_shape}')

            if not is_2d:
                patch_halo = self.predictor_config.get('patch_halo', (0, 0, 0))
                self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
                logger.info(f'Using patch_halo: {patch_halo}')
            
            # Run predictions on the entire input dataset
            start = time.time()
            with torch.no_grad():
                # for batch, indices, _ in tqdm(test_loader, mininterval = 10, desc="gen_predictions"):
                for t in tqdm(test_loader, mininterval = 10, desc="gen_predictions"):
                    if is_2d:
                        for k in included_2d_pl:
                            # send batch to device                            
                            batch, indices, _ = self._split_testing_batch(t[k])
                            if batch is None:
                                continue
                            print(k)
                            predictions = self.model[k](batch)
                            batch_size = predictions.size(0)
                                            
                            # convert to numpy array
                            predictions = predictions.cpu().numpy()
                            if k == "y":
                                # we need to transpose (batch, channel, y, z, x) back to (batch, channel, z, y, x)
                                predictions = np.transpose(predictions, axes=(0, 1, 3, 2, 4))
                            elif k == "x":
                                # we need to transpose (batch, channel, x, z, y) back to (batch, channel, z, y, x)
                                predictions = np.transpose(predictions, axes=(0, 1, 3, 4, 2))
                            # import pdb; pdb.set_trace()
                            for batch_idx in range(batch_size):                                                    
                                # if k == "y":
                                    # import pdb; pdb.set_trace()
                                prediction_map[k][indices[batch_idx]] += predictions[batch_idx, 0, :, :, :]
                                normalization_mask[k][indices[batch_idx]] += 1
                    else:
                        # send batch to device
                        # torch.cuda.synchronize()
                        # start = time.time()
                        # torch.cuda.synchronize()
                        # start = time.time()
                        batch, indices, _ = self._split_testing_batch(t)
                        # end1 = time.time()
                        # print(f"TIME for moving batch = {end1 - start}")
                        predictions, _, _, _, _, _ = self.model(batch)
                        # end2 = time.time()
                        # print(f"TIME for inference = {end2 - end1}")
                        batch_size = predictions.size(0)

                        # import pdb; pdb.set_trace()
                        # (Pdb) pp predictions.shape
                        # torch.Size([32, 1, 64, 64, 64])
                        # convert to numpy array
                        predictions = predictions.cpu().numpy()
                        # import pdb; pdb.set_trace()
                        for batch_idx in range(batch_size):                    
                            prediction_map[indices[batch_idx]] += predictions[batch_idx, 0, :, :, :]
                            normalization_mask[indices[batch_idx]] += 1
            prediction_map = self._save_results_predictions_only(
                prediction_map, 
                normalization_mask, 
                1, 
                h5_output_file, 
                None,
                save_file=save_file,
                is_2d=is_2d,
                included_2d_pl=included_2d_pl,
                included_2d_slices=included_2d_slices,
                fuse_2d_option=fuse_2d_option,
            )
            end2 = time.time()
            print(f"TIME for prediction = {end2 - start}")
            # close the output H5 file
            if h5_output_file is not None:
                h5_output_file.close()
            if is_2d:
                for k in included_2d_pl:
                    self.model[k].testing = False
            else:
                self.model.testing = False
            return prediction_map


    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        label_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        return prediction_maps, normalization_masks, label_maps

    def _save_results_predictions_only(
        self, 
        prediction_map, 
        normalization_mask, 
        output_heads, 
        output_file, 
        dataset, 
        save_file=True,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        fuse_2d_option="mean",
    ):
        """only save predictions h5 files, without evaluating"""
        assert output_heads == 1
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_dataset = self.get_output_dataset_names(output_heads, prefix='predictions')[0]
        if is_2d:
            for k in included_2d_pl:
                prediction_map[k] = prediction_map[k] / normalization_mask[k].clip(min=1e-6)
        else:
            prediction_map = prediction_map / normalization_mask.clip(min=1e-6)
        # if dataset.mirror_padding is not None:
        #     raise NotImplementedError()
        #     z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]
        #     logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')
        #     if is_2d:
        #         for k in included_2d_pl:
        #             prediction_map[k] = prediction_map[k][z_s, y_s, x_s]
        #     else:
        #         prediction_map = prediction_map[z_s, y_s, x_s]
            
        # fuse 2d plane predictions
        if is_2d:
            vol_shape = prediction_map[included_2d_pl[0]].shape
            prediction_map_2d_fused = np.zeros(vol_shape, dtype="float32")
            if fuse_2d_option == "mean":
                for k in included_2d_pl:
                    prediction_map_2d_fused = prediction_map_2d_fused + prediction_map[k]
                prediction_map_2d_fused = prediction_map_2d_fused / len(included_2d_pl)
            else:
                raise NotImplementedError()
            prediction_map = { **prediction_map, **{"fused_2d": prediction_map_2d_fused}}        
        if output_file is not None:
            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")
        return prediction_map

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"