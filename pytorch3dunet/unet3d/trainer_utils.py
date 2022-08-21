"""utils to be used in trainer"""
from typing import AsyncContextManager
import numpy as np
from numpy.lib.arraysetops import isin
import torch
from pytorch3dunet.datasets.standard_dataset import (
    _itk_read_array_from_file,
    _itk_read_image_from_file,
)
import os
import h5py
from pytorch3dunet.datasets.standard_dataset import (
    get_test_loaders_dynamic,
)
from pytorch3dunet.unet3d.predictor import StandardPredictor
from skimage.segmentation import boundaries

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def _gen_spatial_weights_helper(image_array, pred, proxy, algo, existing_label_slices, slice_confidence_neib_radius, fast_gc_seed_map=None):
    if proxy is None or algo is None or algo == "none":
        return None
    if isinstance(algo, float) or isfloat(algo):
        return float(algo)
    if algo.startswith("residule"):
        print(f"algo={algo}")
        return _gen_residule_weights(pred, proxy, algo)
    elif algo.startswith("confidence"):
        print(f"algo={algo}")
        return _gen_confidence_weights(image_array, pred, proxy, existing_label_slices, algo, slice_confidence_neib_radius, fast_gc_seed_map=fast_gc_seed_map)
    else:
        raise NotImplementedError()

def _gen_residule_weights(pred, proxy, algo="residule-dev"):
    """compute residule weights, proxy means proxy label, could be true label, could be proxy label for example by contributor's algorithm"""    
    algo_ = algo.split("-")[1]
    if algo_ == "dev":
        print(f"algo_={algo_}")
        # absolute deviation
        return np.abs(proxy - pred).astype(np.float32)
    elif algo_ == "dice":
        print(f"algo_={algo_}")
        # inverse dice, smaller dice means we need heavier sampling weights for importance training
        return 1.0 - (proxy * pred).astype(np.float32)
    else:
        raise NotImplementedError()

def _gen_confidence_weights(image_array, pred, proxy, existing_label_slices, algo="confidence-dist", slice_confidence_neib_radius=3, fast_gc_seed_map=None):
    """compute confidence weights, note that we provide a existing_label_slices or existing_scribbles as human interactions, the closer to these, the larger the weights"""
    
    # import pdb; pdb.set_trace()
    vol_shape = pred.shape
    vol_shape_map = {"z": vol_shape[0], "y": vol_shape[1], "x": vol_shape[2]}
    
    algo_ = algo.split("-")[1]
    if fast_gc_seed_map is not None:
        assert fast_gc_seed_map.shape == vol_shape
        # in this case, we ONLY use fast_gc_seed_map to create confidence map
        if algo_.startswith("pointreg"):
            print(f"algo_={algo_}")
            point_reg_lambda = float(algo_.split("_")[1])
            # when used as loss weight, do not force the max to be one
            p = np.ones(vol_shape).astype(np.float32)
            p[fast_gc_seed_map > 0] = 1.0 + point_reg_lambda
            return p
        elif algo_.startswith("point_only"):
            print(f"algo_={algo_}")
            p = np.zeros(vol_shape).astype(np.float32)
            p[fast_gc_seed_map > 0] = 1.0
            return p
        elif algo_.startswith("geodesic"):
            raise NotImplementedError()            
        else:
            raise NotImplementedError()

    if not existing_label_slices:
        return None
    if algo_ == "dist":
        print(f"algo_={algo_}")
        # inverse distance, inverse of the minimal distance to three planes
        p_inv_dist = {
            k: [1 for i in range(vol_shape_map[k])] for k in ["z", "y", "x"]
        }
        for k in existing_label_slices.keys():
            if len(existing_label_slices[k]) > 0:
                p_dist_ = [min([abs(x - i) for i in existing_label_slices[k]]) for x in range(vol_shape_map[k])]
                p_inv_dist[k] = [1.0 / (1.0 + x) for x in p_dist_]
        p = np.zeros(vol_shape).astype(np.float32)
        for i in range(vol_shape_map["z"]):
            p[i, :, :] = np.maximum(p[i, :, :], p_inv_dist["z"][i])
        for i in range(vol_shape_map["y"]):
            p[:, i, :] = np.maximum(p[:, i, :], p_inv_dist["y"][i])
        for i in range(vol_shape_map["x"]):
            p[:, :, i] = np.maximum(p[:, :, i], p_inv_dist["x"][i])
        p = p / p.max()
        return p
    elif algo_ == "neib":
        print(f"algo_={algo_}")
        # hard neighbor
        p_inv_dist_nb = {
            k: [1 for i in range(vol_shape_map[k])] for k in ["z", "y", "x"]
        }
        for k in existing_label_slices.keys():
            if len(existing_label_slices[k]) > 0:
                p_dist_ = [min([abs(x - i) for i in existing_label_slices[k]]) for x in range(vol_shape_map[k])]
                p_inv_dist_nb[k] = [float(x <= slice_confidence_neib_radius) for x in p_dist_]
        p = np.ones(vol_shape).astype(np.float32)
        for i in range(vol_shape_map["z"]):
            p[i, :, :] = p[i, :, :] * p_inv_dist_nb["z"][i]
        for i in range(vol_shape_map["y"]):
            p[:, i, :] = p[:, i, :] * p_inv_dist_nb["y"][i]
        for i in range(vol_shape_map["x"]):
            p[:, :, i] = p[:, :, i] * p_inv_dist_nb["x"][i]
        p = p / p.max()
        return p
    elif algo_.startswith("slicereg"):
        slice_reg_lambda = float(algo_.split("_")[1])
        print(f"algo_={algo_}")
        # when used as loss weight, do not force the max to be one
        p = np.ones(vol_shape).astype(np.float32)
        for k in existing_label_slices.keys():
            if len(existing_label_slices[k]) > 0:
                if k == "z":
                    p[existing_label_slices[k], :, :] = 1.0 + slice_reg_lambda
                elif k == "y":
                    p[:, existing_label_slices[k], :] = 1.0 + slice_reg_lambda
                elif k == "x":
                    p[:, :, existing_label_slices[k]] = 1.0 + slice_reg_lambda
                else:
                    raise NotImplementedError()
        return p
    elif algo_ == "slice_only":
        print(f"algo_={algo_}")
        # places where slice labeled : 1.0, elsewhere : 0.0
        # when used as loss weight, do not force the max to be one
        p = np.zeros(vol_shape).astype(np.float32)
        for k in existing_label_slices.keys():
            if len(existing_label_slices[k]) > 0:
                if k == "z":
                    p[existing_label_slices[k], :, :] = 1.0
                elif k == "y":
                    p[:, existing_label_slices[k], :] = 1.0
                elif k == "x":
                    p[:, :, existing_label_slices[k]] = 1.0
                else:
                    raise NotImplementedError()
        return p
    elif algo_ == "geodesic":
        raise NotImplementedError()
    else:
        raise NotImplementedError()

def _gen_spatial_weights(
    image_path,
    pred_path,
    fixed_previous_sample_model_curr_pred,
    label_path,
    propagate_pred_path=None,
    existing_label_slices=None,
    output_spatial_prior_path=None,
    sample_selection_weights_algo="residule-dev",
    loss_weights_algo="confidence-dist",
    proxy_label_weights_algo="confidence-dist",
    slice_confidence_neib_radius=3,
    save_file=False,
    is_2d=False,
    is_use_gt=False,
    existing_fastgc_seed_indices=None,
    initial_fastgc_seed_map=None,
):
    """
        this function provide four types of spatial weights to guide training process
        - 1. spatial weights in 3D to guide MCMC patch sampling (in 2D cases, MCMC sample points, and append z, y, x patch into patch selection) 
            (this could come from either residule OR confidence)
        - 2. spatial weights in 3D to form weights in loss computation (this could come from either residule OR confidence)
        - 3. spatial weights in 3D to form weights in target fusion, target = (1 - weights) * prev_pred + weights * new_label (this should be confidence)

        algo: ["residule", "residule-inv-dist", "inv-dist", "hard-slice-neighbors", "residule-hard-slice-neighbors"]
        output_spatial_prior_path: output h5 file

        is_use_gt: whether to use ground truth to compute spatial weights
    """
    # read label (CANNOT be used directly to compute residule)
    sample_selection_weights = None
    loss_weights = None
    proxy_label_weights = None

    # read pred
    if is_2d:
        raise NotImplementedError()
    else:
        if isinstance(pred_path, str):
            assert os.path.exists(pred_path)
            with h5py.File(pred_path, 'r') as f:
                predictions = f['predictions'][...][:, :, :]
        else:
            assert isinstance(pred_path, np.ndarray)
            predictions = pred_path
    vol_shape = predictions.shape
    vol_shape_map = {"z": vol_shape[0], "y": vol_shape[1], "x": vol_shape[2]}
    sum_axis_map = {"z": (1, 2), "y": (0, 2), "x": (0, 1)}

    # read proxy
    if propagate_pred_path is None:
        if is_use_gt:
            proxy = _itk_read_array_from_file(label_path)
        else:
            proxy = None
    else:
        if isinstance(propagate_pred_path, str):
            proxy = _itk_read_array_from_file(propagate_pred_path)
        else:
            assert isinstance(propagate_pred_path, np.ndarray)
            proxy = propagate_pred_path

    # generate spatial weights
    image_array = _itk_read_array_from_file(image_path)
    sample_selection_weights = _gen_spatial_weights_helper(image_array, predictions, proxy, sample_selection_weights_algo, existing_label_slices, slice_confidence_neib_radius, fast_gc_seed_map=initial_fastgc_seed_map)
    loss_weights = _gen_spatial_weights_helper(image_array, predictions, proxy, loss_weights_algo, existing_label_slices, slice_confidence_neib_radius, fast_gc_seed_map=initial_fastgc_seed_map)
    proxy_label_weights = _gen_spatial_weights_helper(image_array, predictions, proxy, proxy_label_weights_algo, existing_label_slices, slice_confidence_neib_radius, fast_gc_seed_map=initial_fastgc_seed_map)

    if proxy_label_weights is not None and proxy is not None:
        if fixed_previous_sample_model_curr_pred is not None:
            # weighted_proxy_label = (1.0 - proxy_label_weights) * predictions + proxy_label_weights * proxy        
            weighted_proxy_label = (1.0 - proxy_label_weights) * fixed_previous_sample_model_curr_pred + proxy_label_weights * proxy        
        else:
            weighted_proxy_label = proxy
    elif proxy is not None:
        weighted_proxy_label = proxy
    else:
        weighted_proxy_label = None

    if save_file and output_spatial_prior_path is not None:
        h5_output_file = h5py.File(output_spatial_prior_path, 'w')
        h5_output_file.create_dataset("sample_selection_weights", data=sample_selection_weights, compression="gzip")
        h5_output_file.create_dataset("loss_weights", data=loss_weights, compression="gzip")
        h5_output_file.create_dataset("proxy_label_weights", data=proxy_label_weights, compression="gzip")
        h5_output_file.close()
    return sample_selection_weights, loss_weights, proxy_label_weights, weighted_proxy_label

def _read_predictions_heper(pred_path, is_2d, fused_only=False):
    """return torch tensor"""
    if isinstance(pred_path, str):
        assert os.path.exists(pred_path)
        with h5py.File(pred_path, 'r') as f:                
            if is_2d:
                predictions = {k: torch.tensor(f['predictions'][...][k]).squeeze() for k in f['predictions'][...].keys()}
                if fused_only:
                    predictions = predictions["fused_2d"]
            else:
                predictions = torch.tensor(f['predictions'][...][:, :, :]).squeeze()
    else:
        if is_2d:
            assert isinstance(pred_path, dict)
            predictions = {k: torch.tensor(pred_path[k]).squeeze() for k in pred_path.keys()}
        else:                
            assert isinstance(pred_path, np.ndarray)
            predictions = torch.tensor(pred_path).squeeze()
    return predictions

def _gen_evaluations_gt(
    eval_criterion,
    label_path, 
    pred_path,
    is_2d=False,
):
    if isinstance(eval_criterion, dict):
        eval_fn = [v for k, v in eval_criterion.items()][0]  
    else:
        eval_fn = eval_criterion
    # read label
    # read label
    if isinstance(label_path, str):
        true_label = torch.tensor(_itk_read_array_from_file(label_path)).squeeze()
    else:
        assert isinstance(label_path, np.ndarray)
        true_label = torch.tensor(label_path).squeeze()
    # read pred
    predictions = _read_predictions_heper(pred_path, is_2d, fused_only=False)    
    # compute dice
    metrics = {}
    if is_2d:
        for k in predictions.keys():
            metrics["TRUE_CNN_{}".format(k)] = eval_fn(true_label, predictions[k])
    else:
        metrics["TRUE_CNN"] = eval_fn(true_label, predictions)
    return metrics

def _gen_num_slices_gt_label_adhoc(
    label_path,
):
    # read label
    if isinstance(label_path, str):
        true_label = _itk_read_array_from_file(label_path)
    else:
        assert isinstance(label_path, np.ndarray)
        true_label = label_path
    res = {}
    true_label_z = true_label.sum(axis=(1, 2))
    true_label_y = true_label.sum(axis=(0, 2))
    true_label_x = true_label.sum(axis=(0, 1))
    res["z"] = (true_label_z > 0).sum()
    res["y"] = (true_label_y > 0).sum()
    res["x"] = (true_label_x > 0).sum()
    return res

def _gen_evaluations(
    eval_criterion,
    label_path,    
    pred_path,
    propagate_pred_path=None,
    existing_label_slices=None,
    is_2d=False,
):
    """
        new_data_path: nrrd
        new_label_path, true label, nrrd
        existing_label_slices: List[int]
        pred_path: h5
        propagate_pred_path: nrrd
    """
    
    if propagate_pred_path is None:
        return _gen_evaluations_gt(
            eval_criterion,
            label_path, 
            pred_path,
            is_2d=is_2d,
        )
    if isinstance(eval_criterion, dict):
        eval_fn = [v for k, v in eval_criterion.items()][0]  
    else:
        eval_fn = eval_criterion
    # read label
    if isinstance(label_path, str):
        true_label = torch.tensor(_itk_read_array_from_file(label_path)).squeeze()
    else:
        assert isinstance(label_path, np.ndarray)
        true_label = torch.tensor(label_path).squeeze()
    # read pred
    predictions = _read_predictions_heper(pred_path, is_2d, fused_only=True)
    # read contributor pred
    if isinstance(propagate_pred_path, str):
        assert os.path.exists(propagate_pred_path)
        proxy_label = torch.tensor(_itk_read_array_from_file(propagate_pred_path)).squeeze()
    else:
        assert isinstance(propagate_pred_path, np.ndarray)
        proxy_label = torch.tensor(propagate_pred_path).squeeze()

    true_label_slices = {}
    predictions_slices = {}
    proxy_label_slices = {}
    if existing_label_slices:
        for k, v in existing_label_slices.items():
            if not v:
                continue
            if k == "z":
                true_label_slices[k] = true_label[v, :, :]
                predictions_slices[k] = predictions[v, :, :]
                proxy_label_slices[k] = proxy_label[v, :, :]
            elif k == "y":
                true_label_slices[k] = true_label[:, v, :]
                predictions_slices[k] = predictions[:, v, :]
                proxy_label_slices[k] = proxy_label[:, v, :]
            elif k == "x":
                true_label_slices[k] = true_label[:, :, v]
                predictions_slices[k] = predictions[:, :, v]
                proxy_label_slices[k] = proxy_label[:, :, v]
            else:
                raise NotImplementedError()
    
    metrics = {}    
    metrics["TRUE_CNN"] = eval_fn(true_label, predictions)
    metrics["TRUE_PROXY"] = eval_fn(true_label, proxy_label)
    metrics["PROXY_CNN"] = eval_fn(proxy_label, predictions)
    if true_label_slices:
        metrics["TRUE_CNN_SLICE"] = {
            k: eval_fn(true_label_slices[k], predictions_slices[k]) for k in true_label_slices.keys()
        }
        metrics["TRUE_PROXY_SLICE"] = {
            k: eval_fn(true_label_slices[k], proxy_label_slices[k]) for k in true_label_slices.keys()
        }
        metrics["PROXY_CNN_SLICE"] = {
            k: eval_fn(proxy_label_slices[k], predictions_slices[k]) for k in proxy_label_slices.keys()
        }
        # measure user labeling labor through boundary points
        slice_label_boundary_points = 0
        true_label_array = true_label.numpy().astype(np.int32)
        for k, v in existing_label_slices.items():
            if not v:
                continue
            for vv in v:
                if k == "z":
                    slice_label_boundary_points += (true_label_array[vv, :, :] & boundaries.find_boundaries(true_label_array[vv, :, :])).sum()
                elif k == "y":
                    slice_label_boundary_points += (true_label_array[:, vv, :] & boundaries.find_boundaries(true_label_array[:, vv, :])).sum()
                elif k == "x":
                    slice_label_boundary_points += (true_label_array[:, :, vv] & boundaries.find_boundaries(true_label_array[:, :, vv])).sum()
                else:
                    raise NotImplementedError()
        gt_all_boundary_points = 0
        for ii in range(true_label_array.shape[0]):
            gt_all_boundary_points += (true_label_array[ii, :, :] & boundaries.find_boundaries(true_label_array[ii, :, :])).sum()
        metrics["MAX_LABOR_ON_SLICES"] = slice_label_boundary_points
        metrics["GT_LABOR"] = gt_all_boundary_points

    return metrics

def _get_stage(stype, total_samples_seen, sample_id=None, i_effort=None, total_effort=None, existing_label_slices=None, existing_fastgc_seeds=None):
    stage = "{}-sn{}".format(stype, total_samples_seen)
    if sample_id is not None:
        stage += "-s-{}".format(sample_id)
    if i_effort is not None:
        assert total_effort is not None
        stage += "-ie-{}-{}".format(i_effort+1, total_effort)
    if existing_fastgc_seeds is not None and len(existing_fastgc_seeds) > 0:
        stage += "-lpts{}".format(len(existing_fastgc_seeds))
    if existing_label_slices is not None and len(existing_label_slices) > 0:
        for k, v in existing_label_slices.items():
            stage += "-ls{}{}".format(k, len(v))
    return stage


def _gen_predictions(
    model,
    test_config, 
    MODEL_PRED_DIR, 
    new_data_path, 
    output_pred_path, 
    save_file=False,
    is_2d=False,
    included_2d_pl=None,
    included_2d_slices=None,
):
    """
    if is_2d, then return predictions dict
    """
    predictor = StandardPredictor(model=model, output_dir=MODEL_PRED_DIR, config=test_config)

    test_loader_list = []
    for test_loader in get_test_loaders_dynamic(
        config=test_config,
        new_data_path=new_data_path,
        is_2d=is_2d,
        included_2d_pl=included_2d_pl,
        included_2d_slices=included_2d_slices,
        len_option="max",
    ):
        # run the model prediction on the test_loader and save the results in the output_dir
        # predictor(test_loader)
        test_loader_list.append(test_loader)
    assert len(test_loader_list) == 1
    predictions = predictor.gen_predictions(
        test_loader_list, 
        output_pred_path, 
        save_file=save_file,
        is_2d=is_2d,
        included_2d_pl=included_2d_pl,
        included_2d_slices=included_2d_slices,
    )
    return predictions