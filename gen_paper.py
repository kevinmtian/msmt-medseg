"""generate paper submission tables and plots
for eccv
"""
import json
import pandas as pd
import numpy as np
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager

# from paper_data import PAPER_DATA, METHOD_MAPPING, PLOT_MARKERS, PLOT_LINESTYLES

from paper_data_ablation import PAPER_DATA, METHOD_MAPPING, PLOT_MARKERS, PLOT_LINESTYLES

import pickle
import labelme
from labelme import utils
import imgviz
from skimage.segmentation import boundaries
import matplotlib.patches as mpatches
from imgviz.label import label2rgb, label_colormap
import SimpleITK as sitk
import cv2
from IPython import display
import PIL
import subprocess

from matplotlib.patches import Rectangle
from gen_paper_utils import (
    cv2_imshow,
    _itk_read_image_from_file,
    _itk_read_array_from_file,
    _itk_gen_image_from_array,
    _itk_resample_img,
    itk_resize_up_,
    _get_dice,
    binarize,
    scale_img_255,
    get_high_res_path,
    show_propose_slice_weight,
    show_image_with_label,
    plt_helper_get_boundary,
    plt_helper_name_transfer,
    plt_helper_select_plane,
    plt_helper_create_label_legend,
    get_slice_label_resize_up,
)



font_mid = font_manager.FontProperties(size=20)

# 1. read statistics stored in log into a json, for each experiment, devided by two seeds: rs and rsTR, save these into a dataframe
# key = experiment (including rs and rsTR), val = {key=round, val= {train: train_json, test: test_json}}
def gen_stat_dict():
    res = OrderedDict()
    for k, val in PAPER_DATA.items():
        rsTR, file_path = val
        res_curr_test = OrderedDict()
        res_curr_train = OrderedDict()

        with open(file_path, "r") as f:
            for line in f:
                if re.search("TEST_SUMMARY", line):
                    samples_seen = int(re.search(r"(sn)(\d+)", line).group(2))
                    line_json_part = line.split("summary_metrics:")[1].strip().replace("\'", "\"").replace("tensor", "").replace("(","").replace(")", "").replace(".,",",").replace(".}", "}")
                    metrics_dict = json.loads(line_json_part)
                    res_curr_test[samples_seen] = {rsTR: metrics_dict}
                elif re.search("EVAL_SAMPLE_SCORE", line):
                    samples_seen = int(re.search(r"(sn)(\d+)", line).group(2))
                    line_json_part = line.split("sample_eval_score =")[1].strip().replace("\'", "\"").replace("tensor", "").replace("(","").replace(")", "").replace(".,",",").replace(".}", "}")
                    metrics_dict = json.loads(line_json_part)
                    res_curr_train[samples_seen] = {rsTR: metrics_dict}
                else:
                    continue
        res[k] = {
            "test": res_curr_test,
            "train": res_curr_train,
        }    
    return res

# 2. generate csv table from the statistics, for each setting, average/std/min/max of windowed test scores
def gen_stat_table(stat_res, start_point=0, window_size=10, rsTR_list = [1]):
    res_running_avg = OrderedDict()
    res_running_min = OrderedDict()
    for k, v in stat_res.items():
        res_running_avg[k] = {
            "train": OrderedDict(),
            "test": OrderedDict(),
        }
        res_running_min[k] = {
            "train": OrderedDict(),
            "test": OrderedDict(),
        }
        # train
        train_sn = sorted(x for x in v["train"].keys())
        for rsTR in rsTR_list:
            train_score_list = [v["train"][x][rsTR]["TRUE_CNN"] for x in train_sn]
            for i in range(start_point, len(train_sn)):
                if i + window_size > len(train_sn):
                    break
                running_avg = np.mean(train_score_list[i : i + window_size])
                running_min = np.min(train_score_list[i : i + window_size])
                running_key = "sn{}-sn{}".format(train_sn[i], train_sn[i + window_size - 1])
                if running_key not in res_running_avg[k]["train"]:
                    res_running_avg[k]["train"][running_key] = OrderedDict()
                if running_key not in res_running_min[k]["train"]:
                    res_running_min[k]["train"][running_key] = OrderedDict()
                    
                res_running_avg[k]["train"][running_key].update({rsTR: running_avg})
                res_running_min[k]["train"][running_key].update({rsTR: running_min})
        # test
        test_sn = sorted(x for x in v["test"].keys())
        for rsTR in rsTR_list:
            test_score_list = [v["test"][x][rsTR]["TRUE_CNN"]["mean"] for x in test_sn]
            for i in range(start_point, len(test_sn)):
                if i + window_size > len(test_sn):
                    break
                running_avg = np.mean(test_score_list[i : i + window_size])
                running_min = np.min(test_score_list[i : i + window_size])
                running_key = "sn{}-sn{}".format(test_sn[i], test_sn[i + window_size - 1])
                if running_key not in res_running_avg[k]["test"]:
                    res_running_avg[k]["test"][running_key] = OrderedDict()
                if running_key not in res_running_min[k]["test"]:
                    res_running_min[k]["test"][running_key] = OrderedDict()
                    
                res_running_avg[k]["test"][running_key].update({rsTR: running_avg})
                res_running_min[k]["test"][running_key].update({rsTR: running_min})
        
    # remove rsTR pivot by taking the average, std, min and max
    agg_running_avg = OrderedDict()
    agg_running_min = OrderedDict()
    for k, v in res_running_avg.items():
        agg_running_avg[k] = {
            "train": {
                running_key: 
                {
                    "mean": np.mean([val for _, val in v["train"][running_key].items()]),
                    "std": np.std([val for _, val in v["train"][running_key].items()]),
                    "max": np.max([val for _, val in v["train"][running_key].items()]),
                    "min": np.min([val for _, val in v["train"][running_key].items()]),
                } for running_key in v["train"].keys()
            },
            "test": {
                running_key: 
                {
                    "mean": np.mean([val for _, val in v["test"][running_key].items()]),
                    "std": np.std([val for _, val in v["test"][running_key].items()]),
                    "max": np.max([val for _, val in v["test"][running_key].items()]),
                    "min": np.min([val for _, val in v["test"][running_key].items()]),
                } for running_key in v["test"].keys()
            },
        }
    for k, v in res_running_min.items():
        agg_running_min[k] = {
            "train": {
                running_key: 
                {
                    "mean": np.mean([val for _, val in v["train"][running_key].items()]),
                    "std": np.std([val for _, val in v["train"][running_key].items()]),
                    "max": np.max([val for _, val in v["train"][running_key].items()]),
                    "min": np.min([val for _, val in v["train"][running_key].items()]),
                } for running_key in v["train"].keys()
            },
            "test": {
                running_key: 
                {
                    "mean": np.mean([val for _, val in v["test"][running_key].items()]),
                    "std": np.std([val for _, val in v["test"][running_key].items()]),
                    "max": np.max([val for _, val in v["test"][running_key].items()]),
                    "min": np.min([val for _, val in v["test"][running_key].items()]),
                } for running_key in v["test"].keys()
            },
        }

    return res_running_avg, res_running_min, agg_running_avg, agg_running_min

# 3. generate plot
def gen_plot(agg_running_stat, method_name_mapping, start_point=20, end_point=70, data_part="test", stride=1):
    """
    method_name_mapping: {exp_key: display method}    
    """
    all_methods = [x for x in method_name_mapping.keys()]
    sn_dict = {
        int(x.split("-sn")[1]) : x for x in agg_running_stat[all_methods[0]][data_part].keys()
    }
    for method_key, method_name in method_name_mapping.items():
        end_point = min(end_point, max([int(k.split("-sn")[1]) for k in agg_running_stat[method_key][data_part].keys()]))

    sn_dict = {k : v for k, v in sn_dict.items() if k >= start_point and k <= end_point}
    res_return = {}
    
    x_label = sorted([x for x in sn_dict.keys()])

    n_colors = len(method_name_mapping)
    assert n_colors <= 8
    clrs = sns.color_palette("husl", n_colors) # legend colors
    # clrs = sns.color_palette("pastel", n_colors) # legend colors
    linestypes = PLOT_LINESTYLES
    markers = PLOT_MARKERS
    color_dict = {
        k : i for i, k in enumerate(method_name_mapping.keys())
    }
    # create panel
    fig, ax = plt.subplots(1, 1, figsize=(16, 16)) # kip plates
    # plot
    all_handles = []
    all_labels = []
    with sns.axes_style("darkgrid"):
        # draw on panel
        for method_key, method_name in method_name_mapping.items():
            y_mean = [
                agg_running_stat[method_key][data_part][sn_dict[xx]]["mean"] for xx in x_label
            ]
            y_std = [
                agg_running_stat[method_key][data_part][sn_dict[xx]]["std"] for xx in x_label
            ]
            res_return[method_key] = {
                "method_name": method_name,
                "sn": [sn_dict[t] for t in x_label[-1::-5]][:5][::-1],
                "score": [np.round(x, 2) for x in y_mean[-1::-5][:5][::-1]]
            }
            # import pdb; pdb.set_trace()
            # print(f"{method_key}-{y_mean[-1]}")

            ax.plot(x_label[::stride], y_mean[::stride], c=clrs[color_dict[method_key]], linestyle=linestypes[color_dict[method_key]], marker=markers[color_dict[method_key]], 
                markersize=12, label=method_name)
            ax.fill_between(
                x_label[::stride],
                np.array(y_mean[::stride]) - np.array(y_std[::stride]),
                np.array(y_mean[::stride]) + np.array(y_std[::stride]),
                alpha=0.3,
                facecolor=clrs[color_dict[method_key]],
            )
            ax.set_xticks(x_label[::stride], x_label[::stride])
            # ax[p_coor].set_ylabel(name_mapping[k], font=font_zh_big)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_box_aspect(0.8)
            # ax[0].set_title("模型" + name_mapping[k] + " 和 " + label_external_zh, font=font_zh_big)
            handles_, labels_ = ax.get_legend_handles_labels()
            print(handles_, labels_)
            for h_, l_ in zip(handles_, labels_):
                if l_ in all_labels:
                    continue
                all_handles.append(h_)
                all_labels.append(l_)
            
    # plt.subplots_adjust(right=0.8, hspace=0.5)
    plt.subplots_adjust(bottom=0.4, wspace=0.8, hspace=0.8)
    ax.legend(handles=all_handles, labels=all_labels, loc='upper center', bbox_to_anchor=(1.3, 1.0), prop=font_mid, fancybox=False, shadow=False, ncol=1)
    # save and show figure
    plt.savefig(os.path.join("/home/<USERNAME>/Projects/pytorch-3dunet-dev/gen_paper_figure", "figure.png"), bbox_inches='tight', dpi=1024)
    # plt.show()
    df_return = pd.DataFrame(
        {
            "method": [res_return[k]["method_name"] for k in res_return.keys()],
            "sn": [res_return[k]["sn"] for k in res_return.keys()],
            "score": [res_return[k]["score"] for k in res_return.keys()],
        
        }
    )
    df_return.to_csv(os.path.join("/home/<USERNAME>/Projects/pytorch-3dunet-dev/gen_paper_figure", "figure.csv"), header=True, index=False)
    return df_return

if __name__ == "__main__":
    res = gen_stat_dict()
    
    # res_running_avg, res_running_min, agg_running_avg, agg_running_min = gen_stat_table(res, start_point=0, window_size=15)

    # res_return = gen_plot(agg_running_stat=agg_running_avg, method_name_mapping=METHOD_MAPPING, start_point=40, end_point=117, data_part="test")


    res_running_avg, res_running_min, agg_running_avg, agg_running_min = gen_stat_table(res, start_point=0, window_size=15)

    # for brats
    # res_return = gen_plot(agg_running_stat=agg_running_avg, method_name_mapping=METHOD_MAPPING, start_point=40, end_point=117, data_part="test", stride=5)
    # for nci
    res_return = gen_plot(agg_running_stat=agg_running_avg, method_name_mapping=METHOD_MAPPING, start_point=30, end_point=64, data_part="test", stride=5)

    print(res_return)


    print(f"finished")




