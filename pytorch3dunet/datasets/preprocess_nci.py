"""preprocess the nci-isbi2013 data, the preprocessing steps follows identically to the paper of
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

DATA_ROOT = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013"
OUTPUT_PATH = "{}/all".format(DATA_ROOT)

input_path_dict = {}
input_file_list_dict = {}
label_path_dict = {}
label_file_list_dict = {}
all_sources = ["train60_3t", "train60_diagnosis", "lb10_3t", "lb10_diagnosis", "test10_3t", "test10_diagnosis"]

input_path_train60_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/train60/manifest-ZqaK9xEy8795217829022780222/Prostate-3T"
input_file_list_train60_3t = [x for x in os.listdir(input_path_train60_3t) if x.startswith("Prostate3T-")]
label_path_train60_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/train60label"
label_file_list_train60_3t = [x for x in os.listdir(label_path_train60_3t) if x.startswith("Prostate3T-")]

input_path_dict["train60_3t"] = input_path_train60_3t
input_file_list_dict["train60_3t"] = input_file_list_train60_3t
label_path_dict["train60_3t"] = label_path_train60_3t
label_file_list_dict["train60_3t"] = label_file_list_train60_3t

input_path_train60_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/train60/manifest-ZqaK9xEy8795217829022780222/PROSTATE-DIAGNOSIS"
input_file_list_train60_diagnosis = [x for x in os.listdir(input_path_train60_diagnosis) if x.startswith("ProstateDx-")]
label_path_train60_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/train60label"
label_file_list_train60_diagnosis = [x for x in os.listdir(label_path_train60_diagnosis) if x.startswith("ProstateDx-")]

input_path_dict["train60_diagnosis"] = input_path_train60_diagnosis
input_file_list_dict["train60_diagnosis"] = input_file_list_train60_diagnosis
label_path_dict["train60_diagnosis"] = label_path_train60_diagnosis
label_file_list_dict["train60_diagnosis"] = label_file_list_train60_diagnosis

input_path_lb10_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/lb10/manifest-7v55qffK2424752658836301389/Prostate-3T"
input_file_list_lb10_3t = [x for x in os.listdir(input_path_lb10_3t) if x.startswith("Prostate3T-")]
label_path_lb10_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/lb10label"
label_file_list_lb10_3t = [x for x in os.listdir(label_path_lb10_3t) if x.startswith("Prostate3T-")]

input_path_dict["lb10_3t"] = input_path_lb10_3t
input_file_list_dict["lb10_3t"] = input_file_list_lb10_3t
label_path_dict["lb10_3t"] = label_path_lb10_3t
label_file_list_dict["lb10_3t"] = label_file_list_lb10_3t

input_path_lb10_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/lb10/manifest-7v55qffK2424752658836301389/PROSTATE-DIAGNOSIS"
input_file_list_lb10_diagnosis = [x for x in os.listdir(input_path_lb10_diagnosis) if x.startswith("ProstateDx-")]
label_path_lb10_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/lb10label"
label_file_list_lb10_diagnosis = [x for x in os.listdir(label_path_lb10_diagnosis) if x.startswith("ProstateDx-")]

input_path_dict["lb10_diagnosis"] = input_path_lb10_diagnosis
input_file_list_dict["lb10_diagnosis"] = input_file_list_lb10_diagnosis
label_path_dict["lb10_diagnosis"] = label_path_lb10_diagnosis
label_file_list_dict["lb10_diagnosis"] = label_file_list_lb10_diagnosis

input_path_test10_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/test10/manifest-WTWyB8IJ8830296727402453766/Prostate-3T"
input_file_list_test10_3t = [x for x in os.listdir(input_path_test10_3t) if x.startswith("Prostate3T-")]
label_path_test10_3t = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/test10label"
label_file_list_test10_3t = [x for x in os.listdir(label_path_test10_3t) if x.startswith("Prostate3T-")]

input_path_dict["test10_3t"] = input_path_test10_3t
input_file_list_dict["test10_3t"] = input_file_list_test10_3t
label_path_dict["test10_3t"] = label_path_test10_3t
label_file_list_dict["test10_3t"] = label_file_list_test10_3t

input_path_test10_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/test10/manifest-WTWyB8IJ8830296727402453766/PROSTATE-DIAGNOSIS"
input_file_list_test10_diagnosis = [x for x in os.listdir(input_path_test10_diagnosis) if x.startswith("ProstateDx-")]
label_path_test10_diagnosis = "/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013/test10label"
label_file_list_test10_diagnosis = [x for x in os.listdir(label_path_test10_diagnosis) if x.startswith("ProstateDx-")]

input_path_dict["test10_diagnosis"] = input_path_test10_diagnosis
input_file_list_dict["test10_diagnosis"] = input_file_list_test10_diagnosis
label_path_dict["test10_diagnosis"] = label_path_test10_diagnosis
label_file_list_dict["test10_diagnosis"] = label_file_list_test10_diagnosis

def read_dcm_image(
    dicom_dir,    
):
    """read dcm series and save them to 3D images in nrrd"""
    print("Reading Dicom directory:", dicom_dir)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])
    # print("Writing image:", output_path)
    # sitk.WriteImage(image, output_path)
    return image

def get_full_dcm_path(curr_dir):
    x = os.listdir(curr_dir)[0]
    if x.endswith(".dcm"):
        return curr_dir
    else:
        return get_full_dcm_path(os.path.join(curr_dir, x))

def find_and_convert_to_nrrd_and_rename():
    sample_id_dict = {}

    # find and collect
    for source in all_sources:
        # collect image names
        for image_serie in input_file_list_dict[source]:
            orig_serie_dir = os.path.join(input_path_dict[source], image_serie)
            final_serie_dir = get_full_dcm_path(orig_serie_dir)
            # find corresponding label name
            label_name = None
            for ln in label_file_list_dict[source]:
                if ln.startswith(image_serie):
                    label_name = ln
                    break
            if label_name is None:
                raise ValueError(f"label not found for {source} - {image_serie}")
            
            sample_id = "_".join(image_serie.strip().split("-"))
            sample_id_dict[sample_id] = {}
            sample_id_dict[sample_id]["orig_image_path"] = final_serie_dir
            sample_id_dict[sample_id]["orig_label_path"] = os.path.join(label_path_dict[source], label_name)

    # convert image to nrrd and copy and paste to new directory
    for sample_id in sample_id_dict.keys():
        final_serie_dir = sample_id_dict[sample_id]["orig_image_path"]
        orig_label_path = sample_id_dict[sample_id]["orig_label_path"]
        itk_image = read_dcm_image(final_serie_dir)
        label_image = _itk_read_image_from_file(orig_label_path)
        label_array = sitk.GetArrayFromImage(label_image)
        if (label_array < 0).sum() > 0 or ((label_array > 0) & (label_array < 1)).sum() > 0:
            raise RuntimeError(f"label {orig_label_path} has negative values, check!")
        new_label_name = "sample_{}_label.nrrd".format(sample_id)
        shutil.copyfile(sample_id_dict[sample_id]["orig_label_path"], os.path.join(OUTPUT_PATH, new_label_name))
        sample_id_dict[sample_id]["label_path"] = os.path.join(OUTPUT_PATH, new_label_name)

        # binarize label
        binary_label_path = os.path.join(OUTPUT_PATH, "sample_{}_label_binary.nrrd".format(sample_id))
        label_array_binary = np.zeros(label_array.shape).astype(np.float32)
        label_array_binary[label_array > 0] = 1.0
        label_binary = sitk.GetImageFromArray(label_array_binary)
        label_binary.SetSpacing(label_image.GetSpacing())
        label_binary.SetOrigin(label_image.GetOrigin())
        label_binary.SetDirection(label_image.GetDirection())
        _itk_write_image_to_file(label_binary, binary_label_path)
        sample_id_dict[sample_id]["binary_label_path"] = binary_label_path

        # make sure image is aligned with given label
        itk_image.SetSpacing(label_image.GetSpacing())
        itk_image.SetOrigin(label_image.GetOrigin())
        itk_image.SetDirection(label_image.GetDirection())
        # save
        new_image_name = "sample_{}_image.nrrd".format(sample_id)
        _itk_write_image_to_file(itk_image, os.path.join(OUTPUT_PATH, new_image_name))
        sample_id_dict[sample_id]["image_path"] = os.path.join(OUTPUT_PATH, new_image_name)        

    assert len(sample_id_dict) == 80
    return sample_id_dict

def _compute_global_mean_std(sample_id_dict, remove_bg=True):
    count = 0
    mean = 0
    M2 = 0
    existingAggregate = (count, mean, M2)
    for sample_id, v in sample_id_dict.items():        
        image_array = _itk_read_array_from_file(v["image_path"])
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
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["image_path"])
        image_array = sitk.GetArrayFromImage(image)
        normalized_image_array = _itensity_normalize_one_volume_given_stats(image_array, global_mean, global_std, add_noise_to_zero=add_noise_to_zero)
        normalized_image = sitk.GetImageFromArray(normalized_image_array)
        normalized_image.SetSpacing(image.GetSpacing())
        normalized_image.SetOrigin(image.GetOrigin())
        normalized_image.SetDirection(image.GetDirection())
        normalized_image_path = sample_id_dict[sample_id]["image_path"].split(".nrrd")[0] + "_norm.nrrd"
        sample_id_dict[sample_id]["norm_image_path"] = normalized_image_path
        _itk_write_image_to_file(normalized_image, normalized_image_path)
    return sample_id_dict

def crop_images(sample_id_dict, margin=10):
    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["norm_image_path"])        
        image_array = sitk.GetArrayFromImage(image)
        label_array = _itk_read_array_from_file(sample_id_dict[sample_id]["binary_label_path"])
        idx_min, idx_max = _get_ND_bounding_box(label_array, margin=margin)
        image_array_crop = _crop_ND_volume_with_bounding_box(image_array, idx_min, idx_max)
        label_array_crop = _crop_ND_volume_with_bounding_box(label_array, idx_min, idx_max)

        image_crop = sitk.GetImageFromArray(image_array_crop)
        image_crop.SetSpacing(image.GetSpacing())
        image_crop.SetOrigin(image.GetOrigin())
        image_crop.SetDirection(image.GetDirection())
        image_crop_path = sample_id_dict[sample_id]["norm_image_path"].split(".nrrd")[0] + "_crop.nrrd"
        sample_id_dict[sample_id]["image_crop_path"] = image_crop_path
        _itk_write_image_to_file(image_crop, image_crop_path)

        label_crop = sitk.GetImageFromArray(label_array_crop)
        label_crop.SetSpacing(image.GetSpacing())
        label_crop.SetOrigin(image.GetOrigin())
        label_crop.SetDirection(image.GetDirection())
        label_crop_path = sample_id_dict[sample_id]["binary_label_path"].split(".nrrd")[0] + "_crop.nrrd"
        sample_id_dict[sample_id]["label_crop_path"] = label_crop_path
        _itk_write_image_to_file(label_crop, label_crop_path)

    return sample_id_dict

def resize_images(sample_id_dict, out_size=(55, 55, 30)):

    all_sample_ids = sorted(list(sample_id_dict.keys()))
    for i, sample_id in enumerate(all_sample_ids):
        print(i, len(all_sample_ids), sample_id)
        image = _itk_read_image_from_file(sample_id_dict[sample_id]["image_crop_path"])        
        label = _itk_read_image_from_file(sample_id_dict[sample_id]["label_crop_path"])

        image_resize = _itk_resample_img(image, out_size=out_size, is_label=False)
        image_resize.SetSpacing(image.GetSpacing())
        image_resize.SetOrigin(image.GetOrigin())
        image_resize.SetDirection(image.GetDirection())
        image_resize_path = sample_id_dict[sample_id]["image_crop_path"].split(".nrrd")[0] + "_resize.nrrd"
        sample_id_dict[sample_id]["image_resize_path"] = image_resize_path
        _itk_write_image_to_file(image_resize, image_resize_path)

        label_resize = _itk_resample_img(label, out_size=out_size, is_label=True)
        label_resize.SetSpacing(label.GetSpacing())
        label_resize.SetOrigin(label.GetOrigin())
        label_resize.SetDirection(label.GetDirection())
        label_resize_path = sample_id_dict[sample_id]["label_crop_path"].split(".nrrd")[0] + "_resize.nrrd"
        sample_id_dict[sample_id]["label_resize_path"] = label_resize_path
        _itk_write_image_to_file(label_resize, label_resize_path)

    return sample_id_dict

def main():
    # 1. find and rename file name and corresponding label file
    print("==========find_and_rename===========")
    sample_id_dict = find_and_convert_to_nrrd_and_rename()
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)

    # 4. rescale and normalize 
    print("==========normalize_images===========")
    sample_id_dict = normalize_images(sample_id_dict, remove_bg=False, add_noise_to_zero=False)
    print("==========save meta===========")
    with open(os.path.join(OUTPUT_PATH, "sample_id_dict.json"), "w") as f:
        json.dump(sample_id_dict, f)
    
    # 5. crop the image and label according to BB, save as seperate cropped files with an extension of 10 voxels as the margin.
    print("==========crop_images===========")
    sample_id_dict = crop_images(sample_id_dict, margin=10)
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