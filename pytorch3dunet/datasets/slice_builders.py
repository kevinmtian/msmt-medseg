import collections
import importlib
from itertools import permutations
from os import replace
import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import StratifiedShuffleSplit

from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('Dataset')

class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(
        self,
        vol_shape,
        patch_shape,
        stride_shape,
        label,
        is_2d=False,
        included_2d_pl=None,
        included_2d_slices=None,
        threshold=0.6, 
        slack_acceptance=0.01, 
        threshold_count=0,
        max_sample_size=None,
        mcmc_spatial_prior=None,
        mcmc_chain_length=0,
        mcmc_sample_size=0,
        **kwargs
    ):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """
        if label is not None:
            assert isinstance(label, np.ndarray)

        vol_shape = tuple(vol_shape)
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)


        if max_sample_size is None or max_sample_size > 0:
            self._raw_slices = self._build_slices(vol_shape, patch_shape, stride_shape, is_2d, included_2d_pl, included_2d_slices)

            # import pdb; pdb.set_trace()
            
            if label is not None:
                # filter based on ground truth label
                rand_state = np.random.RandomState(47)
                def _ignore_predicate(label_idx):
                    patch = np.copy(label[label_idx])
                    # using patch.sum() could take care of soft labels
                    non_ignore_counts = patch.sum()
                    non_ignore_ratio = patch.sum() / patch.size
                    return non_ignore_counts > threshold_count or non_ignore_ratio > threshold or rand_state.rand() < slack_acceptance
                
                if threshold > 0 or threshold_count > 0:
                    assert slack_acceptance >= 0
                    if is_2d:
                        for k in included_2d_pl:
                            self._raw_slices[k] = list(filter(_ignore_predicate, self._raw_slices[k]))
                    else:
                        self._raw_slices = list(filter(_ignore_predicate, self._raw_slices))

            
            # filter based on random sampling            
            if max_sample_size is not None and max_sample_size > 0:
                if is_2d:
                    if max_sample_size < sum([len(v) for _, v in self._raw_slices.items()]):
                        idx_map = {}
                        curr_count = 0
                        curr_k = 0
                        y = []
                        for k, v in self._raw_slices.items():
                            for i, _ in enumerate(v):
                                idx_map[i + curr_count] = (k, i)
                                y.append(curr_k)
                            curr_count += len(v)
                            curr_k += 1
                        stratified_splitter = StratifiedShuffleSplit(n_splits=2, train_size=max_sample_size, random_state=47)                
                        res_slices = {k: [] for k in self._raw_slices.keys()}
                        for train_index, _ in stratified_splitter.split(np.zeros(curr_count), np.array(y)):
                            for x in train_index:
                                k, i = idx_map[x]
                                res_slices[k].append(self._raw_slices[k][i])
                            break
                        self._raw_slices = res_slices
                else:
                    if max_sample_size < len(self._raw_slices):
                        sample_idx = np.random.choice(len(self._raw_slices), max_sample_size, replace=False)
                        self._raw_slices = [self._raw_slices[i] for i in sample_idx]
        
        # append mcmc sampling slices
        if mcmc_sample_size > 0:
            assert isinstance(mcmc_spatial_prior, np.ndarray)
            if mcmc_spatial_prior.sum() > 0:
                # get candidate patch seeds generated from mcmc sampling
                print("======= start mcmc sampling =======")                    
                if is_2d:
                    mcmc_patch_seeds = {}
                    for k in included_2d_pl:
                        mcmc_patch_seeds_ = {}
                        mcmc_patch_seeds[k] = []
                        for i in included_2d_slices[k]:
                            if k == "z":
                                p_theory = mcmc_spatial_prior[i, :, :]
                                patch_shape_2d = (patch_shape[1], patch_shape[2])
                            elif k == "y":
                                p_theory = mcmc_spatial_prior[:, i, :]
                                patch_shape_2d = (patch_shape[0], patch_shape[2])
                            elif k == "x":
                                p_theory = mcmc_spatial_prior[:, :, i]
                                patch_shape_2d = (patch_shape[0], patch_shape[1])
                            else:
                                raise NotImplementedError()
                            if p_theory.sum() > 0:
                                if p_theory.sum() != 1.0:
                                    p_theory = p_theory / p_theory.sum() 
                                    mcmc_patch_seeds_[i] = self._mcmc_mh_sampling_2d(
                                        p_theory, 
                                        mcmc_chain_length, 
                                        mcmc_sample_size, 
                                        patch_shape_2d,
                                    )
                                    if not len(mcmc_patch_seeds_[i]) == mcmc_sample_size:
                                        logger.warn(f"actual mcmc sample size '{len(mcmc_patch_seeds_[i])}' not equal to requested size of {mcmc_sample_size}")
                                for x in mcmc_patch_seeds_[i]:
                                    if k == "z":
                                        mcmc_patch_seeds[k].append((i, x[0], x[1]))
                                    elif k == "y":
                                        mcmc_patch_seeds[k].append((x[0], i, x[1]))
                                    elif k == "x":
                                        mcmc_patch_seeds[k].append((i, x[0], x[1]))
                                    else:
                                        raise NotImplementedError()

                else:            
                    if mcmc_spatial_prior.sum() != 1.0:
                        mcmc_spatial_prior = mcmc_spatial_prior / mcmc_spatial_prior.sum()
                    mcmc_patch_seeds = self._mcmc_mh_sampling(
                        mcmc_spatial_prior, 
                        mcmc_chain_length, 
                        mcmc_sample_size, 
                        patch_shape,
                    )
                    if not len(self.mcmc_patch_seeds) == mcmc_sample_size:
                        logger.warn(f"actual mcmc sample size {len(mcmc_patch_seeds)} not equal to requested size of {mcmc_sample_size}")

                # assert len(self.mcmc_patch_seeds) == mcmc_sample_size
                print("======= finish mcmc sampling ======")
                self._mcmc_slices = self._build_slices_from_seeds(vol_shape, patch_shape, stride_shape, mcmc_patch_seeds, is_2d, included_2d_pl)
                # merge into _raw_slices
                if is_2d:
                    for k in included_2d_pl:
                        self._raw_slices[k] = self._raw_slices[k] + self._mcmc_slices[k]
                else:
                    self._raw_slices = self._raw_slices + self._mcmc_slices

        # clear out duplications
        if is_2d:
            for k in included_2d_pl:
                self._raw_slices[k] = SliceBuilder._unique_slices(self._raw_slices[k])
        else:
            self._raw_slices = SliceBuilder._unique_slices(self._raw_slices)

        # import pdb; pdb.set_trace()
        # print("debug")
                        

    @staticmethod
    def _unique_slices(slices):
        """make slices unique
        slices: List[(slice1, slice2, slice3)]
        """
        unique_slices = []
        unique_slices_hash = set()
        for idx in slices:
            k = "{}_{}_{}_{}_{}_{}".format(
                idx[0].start, idx[0].stop, idx[1].start, idx[1].stop, idx[2].start, idx[2].stop
            )
            if k in unique_slices_hash:
                continue
            unique_slices_hash.add(k)
            unique_slices.append(idx)
        return slices

    @property
    def raw_slices(self):
        return self._raw_slices

    @staticmethod
    def _build_slices(vol_shape, patch_shape, stride_shape, is_2d=False, included_2d_pl=None, included_2d_slices=None):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        def _build_slices_helper(vol_shape, patch_shape, stride_shape, is_2d=False, selected_2d_pl=None, selected_2d_slices=None):
            slices = []
            if len(vol_shape) == 4:
                raise NotImplementedError
            i_z, i_y, i_x = vol_shape

            k_z, k_y, k_x = patch_shape
            s_z, s_y, s_x = stride_shape
            selected_2d_slices_z = None
            selected_2d_slices_y = None
            selected_2d_slices_x = None
            if is_2d:
                if selected_2d_pl == "z":                    
                    k_z = 1
                    s_z = 1
                    selected_2d_slices_z = selected_2d_slices
                elif selected_2d_pl == "y":
                    k_y = 1
                    s_y = 1
                    selected_2d_slices_y = selected_2d_slices
                elif selected_2d_pl == "x":
                    k_x = 1
                    s_x = 1
                    selected_2d_slices_x = selected_2d_slices
                else:
                    raise NotImplementedError()

            z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z, selected_2d_slices_z)
            # import pdb; pdb.set_trace()
            for z in z_steps:
                y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y, selected_2d_slices_y)
                for y in y_steps:
                    x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x, selected_2d_slices_x)
                    for x in x_steps:
                        slice_idx = (
                            slice(z, z + k_z),
                            slice(y, y + k_y),
                            slice(x, x + k_x)
                        )
                        slices.append(slice_idx)
            return slices        
        if is_2d:
            res_slices = {}
            for k in included_2d_pl:
                res_slices[k] = _build_slices_helper(vol_shape, patch_shape, stride_shape, is_2d=is_2d, selected_2d_pl=k, selected_2d_slices=included_2d_slices[k])
            return res_slices
        return _build_slices_helper(vol_shape, patch_shape, stride_shape, is_2d=is_2d, selected_2d_pl=None, selected_2d_slices=None)


    @staticmethod
    def _build_slices_from_seeds(vol_shape, patch_shape, stride_shape, patch_seeds, is_2d=False, included_2d_pl=None):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        patch_seeds: patch centers selected via mcmc

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        def _build_slices_helper(vol_shape, patch_shape, stride_shape, patch_seeds, is_2d=False, selected_2d_pl=None):
            slices = []

            if len(vol_shape) == 4:
                raise NotImplementedError()

            size_z_patch, size_y_patch, size_x_patch = patch_shape

            if is_2d:
                if selected_2d_pl == "z":
                    size_z_patch = 1
                elif selected_2d_pl == "y":
                    size_y_patch = 1
                elif selected_2d_pl == "x":
                    size_x_patch = 1
                else:
                    raise NotImplementedError()

            r_z_0, r_z_1 = SliceBuilder._get_radius(size_z_patch)
            r_y_0, r_y_1 = SliceBuilder._get_radius(size_y_patch)
            r_x_0, r_x_1 = SliceBuilder._get_radius(size_x_patch)


            for seed_coords in patch_seeds:
                z, y, x = seed_coords[0], seed_coords[1], seed_coords[2]
                slice_idx = (
                    slice(z - r_z_0, z + r_z_1),
                    slice(y - r_y_0, y + r_y_1),
                    slice(x - r_x_0, x + r_x_1)
                )
                slices.append(slice_idx)        
            return slices

        if is_2d:
            assert len(included_2d_pl)
            res_slices = {}
            for k in included_2d_pl:
                res_slices[k] = _build_slices_helper(vol_shape, patch_shape, stride_shape, patch_seeds, is_2d=is_2d, selected_2d_pl=k)
            return res_slices
        return _build_slices_helper(vol_shape, patch_shape, stride_shape, patch_seeds, is_2d=is_2d, selected_2d_pl=None)


    @staticmethod
    def _get_radius(x):
        """get lower and upper radius"""
        if x == 0:
            raise ValueError()
        if x == 1:
            return 0, 1
        if x % 2 == 1:
            return x // 2, x - (x // 2)
        return x // 2, x - (x // 2)

    @staticmethod
    def _mcmc_mh_sampling(p_theory, chain_length, sample_size, patch_shape):
        """
        mcmc mh sampling to generate patch seeds
            p_theory: ndarray, same as raw_data, the spatial prior probability
            chain_length: total length of the mcmc chain, default 10000
            sample_size: number of patch seeds extracted from the last part of mcmc chain
            patch_shape: in 2D cases we extract two of them

        return: the returned sample seeds
        """
        # set up proxu distribution
        # covariance
        if sample_size < 1:
            return []
        q_var=100
        mask_convariance_maxtix_ref = [
            [q_var, 0, 0],
            [0, q_var, 0],
            [0, 0, q_var]
        ]
        np.random.seed(42)
        # stores the chain
        result = []
        sampled_patch_seeds = []

        # set up initial sampling point, default to the mid position of the whole image
        # size of input
        size_z_patch, size_y_patch, size_x_patch = patch_shape

        r_z_0, r_z_1 = SliceBuilder._get_radius(size_z_patch)
        r_y_0, r_y_1 = SliceBuilder._get_radius(size_y_patch)
        r_x_0, r_x_1 = SliceBuilder._get_radius(size_x_patch)


        size_z, size_y, size_x = p_theory.shape        
        x_init = [size_z // 2, size_y // 2, size_x // 2]
        init = np.array(x_init).astype(np.uint32)
        result.append(init)
        p = lambda r: p_theory[r[0], r[1], r[2]]
        q = lambda v : multivariate_normal.rvs(mean=v, cov=mask_convariance_maxtix_ref, size=1).astype(np.uint32)

        def _condition(x, r_z_0, r_z_1, r_y_0, r_y_1, r_x_0, r_x_1):
            return (x[0] - r_z_0 < 0 or x[0] + r_z_1 > size_z) or \
                (x[1] - r_y_0 < 0 or x[1] + r_y_1 > size_y) or \
                (x[2] - r_x_0 < 0 or x[2] + r_x_1 > size_x)
        
        for i in range(chain_length):
            curr = result[i]
            xstar = q(curr)
            if _condition(xstar, r_z_0, r_z_1, r_y_0, r_y_1, r_x_0, r_x_1):
                result.append(curr)
            else:
                if p(curr) == 0:
                    alpha = 1
                else:
                    alpha = min(1, p(xstar) / p(curr))
                u=np.random.rand(1)
                if u<alpha:
                    result.append(xstar)
                    sampled_patch_seeds.append(xstar)
                else:
                    result.append(curr)
        if len(sampled_patch_seeds) < sample_size:
            logger.warn("Requested sample_size = {}, got {}, try increasing chain_length.".format(
                    sample_size,
                    len(sampled_patch_seeds)
                )
            )
        return sampled_patch_seeds[-sample_size : ]

    @staticmethod
    def _mcmc_mh_sampling_2d(p_theory, chain_length, sample_size, patch_shape):
        """
        mcmc mh sampling to generate patch seeds
            p_theory: ndarray, same as raw_data, the spatial prior probability
            chain_length: total length of the mcmc chain, default 10000
            sample_size: number of patch seeds extracted from the last part of mcmc chain
            patch_shape: in 2D cases we extract two of them

        return: the returned sample seeds
        """
        # set up proxu distribution
        # covariance
        if sample_size < 1:
            return []
        q_var=100
        mask_convariance_maxtix_ref = [
            [q_var, 0],
            [0, q_var]
        ]
        np.random.seed(42)
        # stores the chain
        result = []
        sampled_patch_seeds = []

        # set up initial sampling point, default to the mid position of the whole image
        # size of input
        size_y_patch, size_x_patch = patch_shape

        r_y_0, r_y_1 = SliceBuilder._get_radius(size_y_patch)
        r_x_0, r_x_1 = SliceBuilder._get_radius(size_x_patch)

        size_y, size_x = p_theory.shape        
        x_init = [size_y // 2, size_x // 2]
        init = np.array(x_init).astype(np.uint32)
        result.append(init)
        p = lambda r: p_theory[r[0], r[1]]
        q = lambda v : multivariate_normal.rvs(mean=v, cov=mask_convariance_maxtix_ref, size=1).astype(np.uint32)

        def _condition(x, r_y_0, r_y_1, r_x_0, r_x_1):
            return (x[1] - r_y_0 < 0 or x[1] + r_y_1 > size_y) or \
                (x[2] - r_x_0 < 0 or x[2] + r_x_1 > size_x)
        
        for i in range(chain_length):
            curr = result[i]
            xstar = q(curr)
            if _condition(xstar, r_y_0, r_y_1, r_x_0, r_x_1):
                result.append(curr)
            else:
                if p(curr) == 0:
                    alpha = 1
                else:
                    alpha = min(1, p(xstar) / p(curr))
                u=np.random.rand(1)
                if u<alpha:
                    result.append(xstar)
                    sampled_patch_seeds.append(xstar)
                else:
                    result.append(curr)
        if len(sampled_patch_seeds) < sample_size:
            logger.warn("Requested sample_size = {}, got {}, try increasing chain_length.".format(
                    sample_size,
                    len(sampled_patch_seeds)
                )
            )
        return sampled_patch_seeds[-sample_size : ]

    @staticmethod
    def _gen_indices(i, k, s, selected_indices=None):
        if selected_indices and len(selected_indices):
            for j in selected_indices:
                yield j
        else:
            assert i >= k, 'Sample size has to be bigger than the patch size'
            for j in range(0, i - k + 1, s):
                yield j
            if j + k < i:
                yield i - k
