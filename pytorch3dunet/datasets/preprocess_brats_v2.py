"""preprocess brats dataset, the preprocessing steps follows identically to the paper of
CVPR RL iteractive segmentation


"""

import os 
import json
from random import sample
import shutil
from pytorch3dunet.datasets.meddata_process_utils import (
    _itk_read_image_from_file,
    _itk_write_image_to_file,
    _itk_read_array_from_file,
    _get_ND_bounding_box,
    _crop_ND_volume_with_bounding_box,
    _running_stats_update,
    _running_stats_finalize,
    _itensity_normalize_one_volume_given_stats,
    _itk_resample_img,
)
import SimpleITK as sitk
import numpy as np

DATA_ROOT = "/data/<USERNAME>/data/dynamic_segmentation/brats2015/train"
OUTPUT_PATH = "{}/all_v2".format(DATA_ROOT)

def find_and_rename():
    save_name_prefix = ""
    sample_id_dict = {}

    # find and collect
    for d in os.listdir(DATA_ROOT):
        if "DS_Store" in d:
            continue
        if not (("HGG" in d) or ("LGG" in d)):
            continue
        # HGG or LGG
        save_name_prefix += str(d).lower()
        for dd in os.listdir(os.path.join(DATA_ROOT, d)):
            if "DS_Store" in dd:
                continue
            image_name = None
            label_name = None
            for ddd in os.listdir(os.path.join(DATA_ROOT, d, dd)):
                if "DS_Store" in ddd:
                    continue
                if "Flair" in ddd:
                    image_name = ddd
                if "OT" in ddd:
                    label_name = ddd

            if (image_name is None) or (label_name is None):
                raise RuntimeError(f"dd: {os.path.join(DATA_ROOT, d, dd)}, image_name: {image_name}, label_name: {label_name}")

            sample_id = image_name.strip().split(".")[-1]
            if sample_id in sample_id_dict:
                raise RuntimeError("sample_id {} already found in {}".format(sample_id,  sample_id_dict[sample_id]))
            sample_id_dict[sample_id] = {}
            sample_id_dict[sample_id]["orig_image_path"] = os.path.join(DATA_ROOT, d, dd, image_name, image_name + ".mha")
            sample_id_dict[sample_id]["orig_label_path"] = os.path.join(DATA_ROOT, d, dd, label_name, label_name + ".mha")
            sample_id_dict[sample_id]["orig_parent_path"] = os.path.join(DATA_ROOT, d, dd)

            assert os.path.exists(sample_id_dict[sample_id]["orig_image_path"]) and os.path.exists(sample_id_dict[sample_id]["orig_label_path"]) \
                and os.path.exists(sample_id_dict[sample_id]["orig_parent_path"])

    assert len(sample_id_dict) == 274
    # import pdb; pdb.set_trace()
    all_sample_ids = sorted(list(sample_id_dict.keys()))
    # rename and save
    for sample_id in all_sample_ids:
        new_image_name = "sample_{}_image.mha".format(sample_id)
        new_label_name = "sample_{}_label.mha".format(sample_id)
        shutil.copyfile(sample_id_dict[sample_id]["orig_image_path"], os.path.join(OUTPUT_PATH, new_image_name))
        shutil.copyfile(sample_id_dict[sample_id]["orig_label_path"], os.path.join(OUTPUT_PATH, new_label_name))
        sample_id_dict[sample_id]["image_path"] = os.path.join(OUTPUT_PATH, new_image_name)
        sample_id_dict[sample_id]["label_path"] = os.path.join(OUTPUT_PATH, new_label_name)
                
    return sample_id_dict

def _check_and_correct(image_path, label_path, correct_label_path, binary_label_path, unify_spacing=False):
    image = _itk_read_image_from_file(image_path)
    label = _itk_read_image_from_file(label_path)
    label_array = sitk.GetArrayFromImage(label)
    if (label_array < 0).sum() > 0 or ((label_array > 0) & (label_array < 1)).sum() > 0:
        raise RuntimeError(f"label {label_path} has negative values, check!")
    label.SetSpacing(image.GetSpacing())
    label.SetOrigin(image.GetOrigin())
    label.SetDirection(image.GetDirection())
    _itk_write_image_to_file(label, correct_label_path)

    label_array_binary = np.zeros(label_array.shape).astype(np.float32)
    label_array_binary[label_array > 0] = 1.0
    label_binary = sitk.GetImageFromArray(label_array_binary)
    label_binary.SetSpacing(image.GetSpacing())
    label_binary.SetOrigin(image.GetOrigin())
    label_binary.SetDirection(image.GetDirection())
    _itk_write_image_to_file(label_binary, binary_label_path)
    
def gen_binary_labels(sample_id_dict):
    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image_path = sample_id_dict[sample_id]["image_path"]
        label_path = sample_id_dict[sample_id]["label_path"]
        correct_label_path = label_path.split(".mha")[0] + "_corrected.mha"
        binary_label_path = label_path.split(".mha")[0] + "_binary.mha"
        _check_and_correct(image_path, label_path, correct_label_path, binary_label_path)
        sample_id_dict[sample_id]["correct_label_path"] = correct_label_path
        sample_id_dict[sample_id]["binary_label_path"] = binary_label_path
    return sample_id_dict

def crop_images(sample_id_dict, margin=10):
    """in v2, we crop not based on labels, but based on nonzero regions of the raw image"""
    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["image_path"])        
        image_array = sitk.GetArrayFromImage(image)
        label_array = _itk_read_array_from_file(sample_id_dict[sample_id]["binary_label_path"])
        idx_min, idx_max = _get_ND_bounding_box(image_array, margin=margin)
        image_array_crop = _crop_ND_volume_with_bounding_box(image_array, idx_min, idx_max)
        label_array_crop = _crop_ND_volume_with_bounding_box(label_array, idx_min, idx_max)

        image_crop = sitk.GetImageFromArray(image_array_crop)
        image_crop.SetSpacing(image.GetSpacing())
        image_crop.SetOrigin(image.GetOrigin())
        image_crop.SetDirection(image.GetDirection())
        image_crop_path = sample_id_dict[sample_id]["image_path"].split(".mha")[0] + "_crop.mha"
        sample_id_dict[sample_id]["image_crop_path"] = image_crop_path
        _itk_write_image_to_file(image_crop, image_crop_path)

        label_crop = sitk.GetImageFromArray(label_array_crop)
        label_crop.SetSpacing(image.GetSpacing())
        label_crop.SetOrigin(image.GetOrigin())
        label_crop.SetDirection(image.GetDirection())
        label_crop_path = sample_id_dict[sample_id]["binary_label_path"].split(".mha")[0] + "_crop.mha"
        sample_id_dict[sample_id]["label_crop_path"] = label_crop_path
        _itk_write_image_to_file(label_crop, label_crop_path)

    return sample_id_dict

def _compute_global_mean_std(sample_id_dict, remove_bg=True):
    count = 0
    mean = 0
    M2 = 0
    existingAggregate = (count, mean, M2)
    for sample_id, v in sample_id_dict.items():        
        image_array = _itk_read_array_from_file(v["image_crop_path"])
        if remove_bg:
            input = image_array[image_array > 0]
        else:
            input = image_array
        for val in input.reshape(-1).tolist():
            existingAggregate = _running_stats_update(existingAggregate, val)
    mean, variance, sampleVariance = _running_stats_finalize(existingAggregate)
    return mean, np.sqrt(variance)

def normalize_images(sample_id_dict, remove_bg=False, add_noise_to_zero=False):
    global_mean, global_std = _compute_global_mean_std(sample_id_dict, remove_bg=remove_bg)
    print(f"Computed global_mean = {global_mean}, global_std = {global_std}")
    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["image_crop_path"])
        image_array = sitk.GetArrayFromImage(image)
        normalized_image_array = _itensity_normalize_one_volume_given_stats(image_array, global_mean, global_std, add_noise_to_zero=add_noise_to_zero)
        normalized_image = sitk.GetImageFromArray(normalized_image_array)
        normalized_image.SetSpacing(image.GetSpacing())
        normalized_image.SetOrigin(image.GetOrigin())
        normalized_image.SetDirection(image.GetDirection())
        normalized_image_path = sample_id_dict[sample_id]["image_crop_path"].split(".mha")[0] + "_norm.mha"
        sample_id_dict[sample_id]["norm_image_path"] = normalized_image_path
        _itk_write_image_to_file(normalized_image, normalized_image_path)
    return sample_id_dict


def resize_images(sample_id_dict, out_size=(55, 55, 30)):

    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["norm_image_path"])        
        label = _itk_read_image_from_file(sample_id_dict[sample_id]["label_crop_path"])

        image_resize = _itk_resample_img(image, out_size=out_size, is_label=False)
        image_resize.SetSpacing(image.GetSpacing())
        image_resize.SetOrigin(image.GetOrigin())
        image_resize.SetDirection(image.GetDirection())
        image_resize_path = sample_id_dict[sample_id]["norm_image_path"].split(".mha")[0] + "_resize.mha"
        sample_id_dict[sample_id]["image_resize_path"] = image_resize_path
        _itk_write_image_to_file(image_resize, image_resize_path)

        label_resize = _itk_resample_img(label, out_size=out_size, is_label=True)
        label_resize.SetSpacing(label.GetSpacing())
        label_resize.SetOrigin(label.GetOrigin())
        label_resize.SetDirection(label.GetDirection())
        label_resize_path = sample_id_dict[sample_id]["label_crop_path"].split(".mha")[0] + "_resize.mha"
        sample_id_dict[sample_id]["label_resize_path"] = label_resize_path
        _itk_write_image_to_file(label_resize, label_resize_path)

    return sample_id_dict

def main():

    # 1. find and rename file name and corresponding label file
    print("==========find_and_rename===========")
    sample_id_dict = find_and_rename()
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)

    # 2. check the unique values of label file, generate binary label with same origin, direction and spacing as training 
    # 3. Optional (make the data spacing to be 1, 1, 1 or anisopic), save as additional files
    print("==========gen_binary_labels===========")
    sample_id_dict = gen_binary_labels(sample_id_dict)
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)
    
    # 5. crop the image and label according to BB, save as seperate cropped files with an extension of 10 voxels as the margin.
    print("==========crop_images===========")
    sample_id_dict = crop_images(sample_id_dict, margin=10)
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)

    # 4. rescale and normalize 
    print("==========normalize_images===========")
    sample_id_dict = normalize_images(sample_id_dict, remove_bg=False, add_noise_to_zero=False)
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)
    
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "r") as f:
        sample_id_dict = json.load(f)
    # 6. resize the image and label to be the size of 50, 50, 30 in their original [x, y, z] nifty directions and save them as training input
    print("==========resize_images===========")
    sample_id_dict = resize_images(sample_id_dict, out_size=(55, 55, 30))
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)

if __name__ == "__main__":
    main()