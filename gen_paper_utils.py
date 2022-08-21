"""generate paper submission tables and plots
for eccv
"""
from unicodedata import numeric
from skimage import io, color

from email.mime import image
import json
import pandas as pd
import numpy as np
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager
from paper_data import PAPER_DATA, METHOD_MAPPING


import pickle

import labelme
from labelme import utils
import imgviz
from skimage.segmentation import boundaries
import matplotlib.patches as mpatches
from imgviz.label_multi import label2rgb, label_colormap
import SimpleITK as sitk
import cv2
from IPython import display
import PIL
import subprocess

from matplotlib.patches import Rectangle

font_mid = font_manager.FontProperties(size=12)

# 4. generate segmentation vis

def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
        (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
        image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))

def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))

def _itk_gen_image_from_array(image_array, ref_image):
    """
    extract left part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        left hippo label, binary mask    
    """
    xxai = sitk.GetImageFromArray(image_array)
    xxai.SetSpacing(ref_image.GetSpacing())
    xxai.SetOrigin(ref_image.GetOrigin())
    xxai.SetDirection(ref_image.GetDirection())    
    return xxai

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

# resampling the (either true or produced) binary label to original resolution! (for better show in the paper)
def itk_resize_up_(input_image_array, ref_itk_input_image_path, ref_itk_ouptut_image_path, is_label=True, is_soft=False):
    """if is_label, then apply nearest neighbor interpolation, else apply linear interpolation"""
    ref_itk_input_image = _itk_read_image_from_file(ref_itk_input_image_path)
    ref_itk_output_image = _itk_read_image_from_file(ref_itk_ouptut_image_path)    
    itk_input_image = _itk_gen_image_from_array(input_image_array, ref_itk_input_image)
    # resize up!
    out_size = ref_itk_output_image.GetSize()
    itk_input_image_resize = _itk_resample_img(itk_input_image, out_size=out_size, is_label=is_label and (not is_soft))
    itk_input_image_resize.SetSpacing(ref_itk_output_image.GetSpacing())
    itk_input_image_resize.SetOrigin(ref_itk_output_image.GetOrigin())
    itk_input_image_resize.SetDirection(ref_itk_output_image.GetDirection())

    assert itk_input_image_resize.GetSize() == ref_itk_output_image.GetSize()
    # convert to array
    input_image_array_resize = sitk.GetArrayFromImage(itk_input_image_resize)
    
    if is_label:
        input_image_array_resize[input_image_array_resize > 1.0] = 0.0
        input_image_array_resize[input_image_array_resize < 0.0] = 0.0
    return input_image_array_resize



def _get_dice(input, target, epsilon=1e-6, weight=None):

    # input and target shapes must match
    assert weight is None
    assert input.shape == target.shape, "'input' and 'target' must have the same shape"

    # input = flatten(input)
    # target = flatten(target)
    input = input.astype(np.float32)
    target = target.astype(np.float32)

    # compute per channel Dice Coefficient
    intersect = (input * target).sum()
    # if weight is not None:
    #     intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = input.sum() + target.sum()
    return 2 * (intersect / np.clip(denominator, a_min=epsilon, a_max=None))

def binarize(img, thres=0.5):    
    img = img.astype(np.float32)
    # try rescale image to 0 1
#     img = (img - img.min()) / (img.max() - img.min()) * 1.0    
    img_b = (img > thres).astype(np.float32)
    return img_b

def scale_img_255(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    return img

def get_high_res_path(path):
    """retrieve high resolution path from resized file path"""
    path_split = path.split("_resize")
    return path_split[0] + path_split[1]
    
    
def show_propose_slice_weight(gt_array, pred_array, plane, plane_slice_num, key=None,save_path=None):
    p_res = np.abs(gt_array - pred_array).astype(np.float32)
    p_res = scale_img_255(p_res)
    img_viz = plt_helper_select_plane(p_res, plane, plane_slice_num)
    
    if key is not None:
        print(key)
    plt.figure(dpi=1200)
    plt.axis('off')
    plt.imshow(img_viz, cmap="coolwarm")
        
    if save_path is not None:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
        
def show_image_with_label(
    image_array, 
    plane, 
    plane_slice_num, 
    label_dict=None, 
    save_path=None, 
    slice_label_dict=None, 
    key=None,
    legend_only=False,
    label_boundary_only=True,    
    label_key_only=None,
):
    """image_array: the highres imagearray
        plane: which plane to cut slices [z, y, x]
        plane_slice_num: which percentile number to cut
        label_dict: dict[label_key, label_array]: should be binary and highres!
        label_boundary_only: if true, draw only label boundary, if false draw label map
        
        return: show image, multiple labels together with legends    
    """    
    if image_array is None and label_dict is None:
        return
    fig = plt.figure(dpi=1200)
    plt.axis('off')
    if image_array is not None:
        image_plane_rgb = imgviz.gray2rgb(
            scale_img_255(
                plt_helper_select_plane(image_array, plane, plane_slice_num),
            ),            
        )
    else:
        image_plane_rgb = None
    if label_dict is None or len(label_dict) == 0:
        # only show image 
        if not legend_only:
            plt.imshow(image_plane_rgb, cmap="gray")
            plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        return

    # extract label boundary
    if label_boundary_only:
        label_boundary_dict = {
            k: plt_helper_get_boundary(plt_helper_select_plane(label_dict[k], plane, plane_slice_num)) for k, _ in label_dict.items()
        }
    else:
        label_boundary_dict = {
            k: plt_helper_get_labelmap(plt_helper_select_plane(label_dict[k], plane, plane_slice_num)) for k, _ in label_dict.items()
        }
    if label_key_only is not None:
        # only plot one type of label (for example, the ground truth, or the model pred)
        assert label_key_only in label_boundary_dict
        label_boundary_dict = {label_key_only : label_boundary_dict[label_key_only]}
    # slice label array
    slice_label_array_dict = {}
    slice_label_array = None
    if slice_label_dict is not None:
        for k, v in slice_label_dict.items():
            if not len(v):
                continue
            if slice_label_array is None:
                slice_label_array = np.zeros(image_array.shape).astype(np.float32)
            if k == "z":
                slice_label_array = np.zeros(image_array.shape).astype(np.float32)
                slice_label_array[v, :, :] = 1.0
                slice_label_array_dict["z"] = plt_helper_select_plane(slice_label_array, plane, plane_slice_num)
                continue
            elif k == "y":
                slice_label_array = np.zeros(image_array.shape).astype(np.float32)
                slice_label_array[:, v, :] = 1.0
                slice_label_array_dict["y"] = plt_helper_select_plane(slice_label_array, plane, plane_slice_num)
                continue
            elif k == "x":
                slice_label_array = np.zeros(image_array.shape).astype(np.float32)
                slice_label_array[:, :, v] = 1.0
                slice_label_array_dict["x"] = plt_helper_select_plane(slice_label_array, plane, plane_slice_num)
                continue
            else:
                raise NotImplementedError()
                
    
    # create label legend    
    selected_planes = set(list(slice_label_array_dict.keys())) - set([plane])
    show_slice_label_dict = {k: slice_label_array_dict[k] for k in selected_planes}
    legend_dict, legend_data, final_label_array, colormap = plt_helper_create_label_legend(
        label_boundary_dict, 
        show_slice_label_dict,
    )
    
    if key is not None:
        print(key)
    
            
    # plot labels and images
    # final_label_arrays stores each type of label in a list, creating overlapped visualization
    # img_viz = label2rgb(
    #     labels = final_label_arrays,
    #     image = image_plane_rgb,
    #     alpha=1.0,
    # )
    # import pdb; pdb.set_trace()
    print(np.unique(final_label_array))
    # final_label_array[final_label_array == 2] = 1
    # final_label_array[final_label_array == 3] = 2
    # final_label_array[final_label_array == 4] = 3
    if label_key_only == "real_gt":
        img_viz = color.label2rgb(final_label_array.astype(np.int32), image_plane_rgb, colors=[colormap[2],],alpha=0.004, bg_label=0,
            bg_color=None, image_alpha=1)
    elif label_key_only == "upsized_curr_pred":
        img_viz = color.label2rgb(final_label_array.astype(np.int32), image_plane_rgb, colors=[colormap[4],],alpha=0.004, bg_label=0, 
            bg_color=None, image_alpha=1)
    else:
        img_viz = color.label2rgb(final_label_array.astype(np.int32), image_plane_rgb, colors=[colormap[2], np.round((colormap[2] + colormap[4]) / 2).astype(np.uint8), colormap[4],],alpha=0.004, bg_label=0, 
            bg_color=None, image_alpha=1)
    img_viz = img_viz.astype(np.float32)
    img_viz = img_viz * 255
    img_viz = np.clip(np.round(img_viz), 0, 255).astype(np.uint8)

    orig_shape = img_viz.shape
#     plt.imshow(img_viz[1 : (orig_shape[0] - 1), 1 : (orig_shape[1] - 1), :], cmap="gray")
    if not legend_only:
        plt.imshow(img_viz, cmap="gray")
            
    # add legend    
    handles = [
        Rectangle((0,0),1,1, color = tuple((v/255 for v in c.tolist()))) for k,c,n in legend_data
    ]
    labels = [n for k,c,n in legend_data]
    if legend_only:
        lgd = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
            fancybox=True, shadow=False, ncol=len(labels))
        save_path = save_path.split(".png")[0] + "-legend" + ".png"
    
    # show and save   
    if save_path is not None:        
        if legend_only:
            fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=1200, bbox_inches='tight')


    # plt.show()

def plt_helper_get_boundary(label):    
    lbl_boundary = (label.astype(int) & boundaries.find_boundaries(label.astype(int))).astype(int)
    return lbl_boundary

def plt_helper_get_labelmap(label):    
    lbl_map = label.astype(int)
    return lbl_map

def plt_helper_name_transfer():
    name_transfer = {
        "real_gt": "Ground Truth",
        "upsized_gt": "upsized_gt",
        "upsized_proxy_label": "Proxy Mask Prediction",
        "upsized_curr_pred": "Segmenter Prediction",
    }
    return name_transfer
    
def plt_helper_select_plane(arr, plane="z", plane_slice_num=16):
    if plane == "z":
        return arr[plane_slice_num, :, :]
    elif plane == "y":
        return arr[:, plane_slice_num, :]
    elif plane == "x":
        return arr[:, :, plane_slice_num]
    else:
        raise NotImplementedError()

def plt_helper_create_label_legend(label_dict, show_slice_label_dict=None):
    legend_dict = {}
    legend_data = []
    # final_label_arrays = []
    
    colormap = label_colormap()
    mult_factor = 1
    counter = 1
    final_label_array = None
    assert len(label_dict) <= 2
    numbers_seen = []
    for k, label_slice_array in label_dict.items():
        if final_label_array is None:
            final_label_array = np.zeros(label_slice_array.shape).astype(np.float32)
        # this will cause newer labels to cover old labels. This would not be a problem when we draw label boundaries only, however,
        # this will be an issue if we want to display overlapped half-transparent labels together.
        # final_label_array += (label_slice_array > 0).astype(np.float32)
        join = (final_label_array > 0) & (label_slice_array > 0)
        extra = (final_label_array == 0) & (label_slice_array > 0)
        if not numbers_seen:
            numbers_seen.append(1)
        final_label_array[join] = numbers_seen[-1]
        numbers_seen.append(numbers_seen[-1] + 1)
        final_label_array[extra] = numbers_seen[-1]
        numbers_seen.append(numbers_seen[-1] + 1)
        if k == "real_gt":
            index = 2
        elif k == "upsized_curr_pred":
            index = 4
        else:
            raise NotImplementedError(k)
        legend_dict[k] = {
            "index": index,
            "rgb": colormap[index, :],
            "name": plt_helper_name_transfer().get(k, k),            
        }
        legend_data.append(
            [
                legend_dict[k]["index"],
                legend_dict[k]["rgb"],
                legend_dict[k]["name"],
            ]        
        )
        # final_label_arrays.append(final_label_array.astype(np.int32))
        counter += 1
    
    # append slice labeling
    if show_slice_label_dict is not None:
        if final_label_array is None:
            final_label_array = np.zeros(show_slice_label_dict[list(show_slice_label_dict.keys())[0]].shape).astype(np.float32)
        for k, slice_label_array in show_slice_label_dict.items():
            final_label_array[slice_label_array > 0] = mult_factor * (counter + 1)
        legend_dict_ = {
            "index": counter,
            "rgb": colormap[int(mult_factor * (counter + counter)), :],
            "name": "labeled slices",
        }
        legend_data.append(
            [
                legend_dict_["index"],
                legend_dict_["rgb"],
                legend_dict_["name"],
            ]
        )        
    return legend_dict, legend_data, final_label_array, colormap

def get_slice_label_resize_up(label_slices, orig_sz, up_sz):
    ratio = float(up_sz) / float(orig_sz)
    up_label_slices = []
    for x in label_slices:
        y = (x * ratio + (x + 1) * ratio) / 2
        y = max(0, min(int(y), up_sz - 1))
        up_label_slices.append(y)
    
    return up_label_slices


