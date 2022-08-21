"""utils that support loading and process medical images"""

import SimpleITK as sitk
import sys
import os
import nibabel
import numpy as np
import random
from scipy import ndimage
import SimpleITK as sitk
import subprocess

# loading and saving utils

# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#

def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    inputs:
        folder_list: a list of folders
        file_name:  filename
    outputs:
        full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
    return full_file_name

def load_3d_volume_as_array(filename):
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))

def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def _running_stats_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def _running_stats_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

def _itensity_normalize_one_volume_given_stats(volume, global_mean, global_std, add_noise_to_zero=False):
    """volume should be the one before taking any normalization"""
    out = (volume - global_mean) / global_std
    if add_noise_to_zero:
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
    return out

def _itensity_normalize_one_volume(volume, remove_bg=False, add_noise_to_zero=False):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
        remove_bg: do not include background pixels when computing mean and std
    outputs:
        out: the normalized nd volume
    """
    if remove_bg:
        pixels = volume[volume > 0]
    else:
        pixels = volume
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    if add_noise_to_zero:
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
    return out

def _get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


def _crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[slice(min_idx[0], max_idx[0] + 1),
                        slice(min_idx[1], max_idx[1] + 1)]
    elif(dim == 3):
        output = volume[slice(min_idx[0], max_idx[0] + 1),
                        slice(min_idx[1], max_idx[1] + 1),
                        slice(min_idx[2], max_idx[2] + 1)]
    elif(dim == 4):
        output = volume[slice(min_idx[0], max_idx[0] + 1),
                        slice(min_idx[1], max_idx[1] + 1),
                        slice(min_idx[2], max_idx[2] + 1),
                        slice(min_idx[3], max_idx[3] + 1)]
    elif(dim == 5):
        output = volume[slice(min_idx[0], max_idx[0] + 1),
                        slice(min_idx[1], max_idx[1] + 1),
                        slice(min_idx[2], max_idx[2] + 1),
                        slice(min_idx[3], max_idx[3] + 1),
                        slice(min_idx[4], max_idx[4] + 1)]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume
        
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center    

def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif(slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif(slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes

def _itk_resample_img(itk_image, out_spacing=None, out_size = (55,55,30), is_label=False, is_c3d=False, c3d_input=None, c3d_output=None):
    if is_c3d:
        raise NotImplementedError()
        
    assert out_spacing is None or out_size is None
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    if out_spacing is not None:
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
        ]

    if out_size is not None:
        out_spacing = [
            np.round(original_spacing[0] * (original_size[0] / out_size[0]), 2),
            np.round(original_spacing[1] * (original_size[1] / out_size[1]), 2),
            np.round(original_spacing[2] * (original_size[2] / out_size[2]), 2)
        ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
    return out_volume

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume  

def get_largest_two_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img

def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component


def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """
    
    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext,s) # labeling
    sizes = ndimage.sum(lab_ext,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli =  np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if((overlap.sum()+ 0.0)/sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice


def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))
    
def _itk_write_image_to_file(itk_image, output_path):
    sitk.WriteImage(itk_image, output_path, True)

def append_dir_prefix(root_dir, file_name):
    return os.path.join(root_dir, file_name)

def append_dir_prefix_list(root_dir, file_name_list):
    return [append_dir_prefix(root_dir, x) for x in file_name_list]

def generate_hippo_image_and_label_paths(data_dir):
    """
    args:
        data_dir: data dir to save image and label
    
    """
    image_path_list = []
    label_path_list = []
    whole_label_path_list = []
    left_label_path_list = []
    right_label_path_list = []
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.startswith("ADNI_") and not ("label" in file_name):
            image_path_list.append(append_dir_prefix(data_dir, file_name))
            label_path_list.append(
                append_dir_prefix(
                    data_dir,
                    "{}-label.nrrd".format(file_name.split(".")[0])
                )
            )
            whole_label_path_list.append(
                append_dir_prefix(
                    data_dir,
                    "{}-whole-label.nrrd".format(file_name.split(".")[0])
                )
            )
            left_label_path_list.append(
                append_dir_prefix(
                    data_dir,
                    "{}-left-label.nrrd".format(file_name.split(".")[0])
                )
            )
            right_label_path_list.append(
                append_dir_prefix(
                    data_dir,
                    "{}-right-label.nrrd".format(file_name.split(".")[0])
                )
            )
    return image_path_list, label_path_list, whole_label_path_list, left_label_path_list, right_label_path_list


def generate_hippo_reg_paths(data_dir, file_path_list):
    """
    return trm, reg, inv_reg
    """
    file_names = [os.path.basename(x) for x in file_path_list]
    transform_path_list = append_dir_prefix_list(data_dir, ["{}-transform.h5".format(x.split(".")[0]) for x in file_names])
    reg_path_list = append_dir_prefix_list(data_dir, ["{}-reg.nrrd".format(x.split(".")[0]) for x in file_names])
    inv_reg_path_list = append_dir_prefix_list(data_dir, ["{}-inv-reg.nrrd".format(x.split(".")[0]) for x in file_names])
    return transform_path_list, reg_path_list, inv_reg_path_list

def get_hippo_label_whole(image_label):
    """
    extract whole part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        right hippo label, binary mask    
    """
    xxa = sitk.GetArrayFromImage(image_label)
    xxa[xxa == 2] = 1.0
    xxai = sitk.GetImageFromArray(xxa)
    xxai.SetSpacing(image_label.GetSpacing())
    xxai.SetOrigin(image_label.GetOrigin())
    xxai.SetDirection(image_label.GetDirection())    
    return xxai

def get_hippo_label_right(image_label):
    """
    extract right part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        right hippo label, binary mask    
    """
    xxa = sitk.GetArrayFromImage(image_label)
    xxa[xxa == 2] = 0.0
    xxai = sitk.GetImageFromArray(xxa)
    xxai.SetSpacing(image_label.GetSpacing())
    xxai.SetOrigin(image_label.GetOrigin())
    xxai.SetDirection(image_label.GetDirection())    
    return xxai

def get_hippo_label_left(image_label):
    """
    extract left part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        left hippo label, binary mask    
    """
    xxa = sitk.GetArrayFromImage(image_label)
    xxa[xxa == 1] = 0.0
    xxa[xxa == 2] = 1.0
    xxai = sitk.GetImageFromArray(xxa)
    xxai.SetSpacing(image_label.GetSpacing())
    xxai.SetOrigin(image_label.GetOrigin())
    xxai.SetDirection(image_label.GetDirection())    
    return xxai

def get_hippo_label_identity(image_label):
    """
    DUMMY
    extract full part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        full hippo label, binary mask    
    """
    xxa = sitk.GetArrayFromImage(image_label)
    xxai = sitk.GetImageFromArray(xxa)
    xxai.SetSpacing(image_label.GetSpacing())
    xxai.SetOrigin(image_label.GetOrigin())
    xxai.SetDirection(image_label.GetDirection())
    
    return xxai


# processing utils (cropping, label unification, etc)


# registration utils
def _itk_command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


def itk_registration(fixed_image, moving_image):
    """
    args:
        fixed_image: itk image float 32
        moving_image: itk image float 32
    return:
        itk_transform: that from moving_image -> fixed_image
    """    
    R = sitk.ImageRegistrationMethod()
    #TODO param to arguments
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8
    )
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Similarity3DTransform()
    )
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: _itk_command_iteration(R))
    outTx = R.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    return outTx


def itk_apply_transform(ref_image, input_image, itk_transform, interpolator=sitk.sitkLinear):
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


def itk_registration_all_images(fixed_image_path="", moving_image_path_list=[], output_transform_path_list=[]):
    """
    args:
        fixed_image_path: str, path
        moving_image_path_list: [str]
        output_transform_path_list: [str]
    saves registered transforms to output_transform_path_list
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    for moving_img_path, output_tfm_path in zip(moving_image_path_list, output_transform_path_list):
        moving_image = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)
        itk_transform = itk_registration(fixed_image, moving_image)
        print("write transform to {}".format(output_tfm_path))
        sitk.WriteTransform(itk_transform, output_tfm_path)

def itk_apply_transform_all_images(fixed_image_path="", input_image_path_list=[], input_transform_path_list=[], output_image_path_list=[], interpolator=sitk.sitkLinear):
    """
    args:
        fixed_image_path: str, path
        input_image_path_list: [str]
        input_transform_path_list: [str]
        output_image_path_list: [str]
    saves transformed images to output_image_path_list
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    for input_img_path, input_tfm_path, output_img_path in zip(input_image_path_list, input_transform_path_list, output_image_path_list):
        input_image = sitk.ReadImage(input_img_path, sitk.sitkFloat32)
        itk_transform = sitk.ReadTransform(input_tfm_path)
        output_image = itk_apply_transform(fixed_image, input_image, itk_transform, interpolator)
        print("write output image to {}".format(output_img_path))
        sitk.WriteImage(output_image, output_img_path, True)


def itk_apply_inverse_transform_all_images(ref_image_path_list=[], input_image_path_list=[], input_transform_path_list=[], output_image_path_list=[], interpolator=sitk.sitkLinear):
    """
    args:
        input_image_path_list: [str]
        input_transform_path_list: [str]
        output_image_path_list: [str]
    saves inverse transformed images to output_image_path_list
    """
    for ref_img_path, input_img_path, input_tfm_path, output_img_path in zip(ref_image_path_list, input_image_path_list, input_transform_path_list, output_image_path_list):
        ref_image = sitk.ReadImage(ref_img_path, sitk.sitkFloat32)
        input_image = sitk.ReadImage(input_img_path, sitk.sitkFloat32)
        itk_transform = sitk.ReadTransform(input_tfm_path).GetInverse()
        output_image = itk_apply_transform(ref_image, input_image, itk_transform, interpolator)
        print("write output image to {}".format(output_img_path))
        sitk.WriteImage(output_image, output_img_path, True)
