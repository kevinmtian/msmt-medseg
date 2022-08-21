
## msmt-incseg

Official implementation of the ECCV-MCV workshop submission "Multi-Scale Multi-Task distillation for incremental 3D medical image segmentation"

## Reference

Part of the implementations were inspired by

- [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
- [UCD](https://github.com/ygjwd12345/UCD)


## Requirements
To install the backbone package `pytorch3dunet`:
```
pip install -e .
```

Additional requirements might include:
```
torch=1.7.1
nibabel
SimpleITK
scipy
numpy
h5py
matplotlib
seaborn
imgviz
skimage
labelme
opencv-python
Pillow
```

## Data Download and Preprocess

#### NCI-ISBI2013 dataset
The NCI challenge dataset can be obtained from the [official website](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) and the [download page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267207)

Place the download folders into a directory structured as `/data/<USERNAME>/data/dynamic_segmentation/nci-isbi2013` (place your <USERNAME> in it), then run the preprocessing code `pytorch3dunet/datasets/preprocess_nci.py` after properly replacing the desensitized token <USERNAME>. 

#### BraTS2015 dataset
The BraTS2015 dataset can be obtained from its [challenge website](https://www.smir.ch/BRATS/Start2015)

Place the download folders into a directory structured as `/data/<USERNAME>/data/dynamic_segmentation/brats2015/train` (place your <USERNAME> in it), then run the preprocessing code `pytorch3dunet/datasets/preprocess_brats_v2.py` after properly replacing the desensitized token <USERNAME>. 

## Training and Inference
Training scripts corresponding to Tables 1 and 2 in the paper are provided under `scripts/`
- For NCI-ISBI2013 dataset, look into `scripts/eccv_nci/*.sh` 
- For BraTS2015 dataset, look into `scripts/eccv_brats/*.sh` 

The file name is corresponding to items in the table, for example, to reproduce training for `Mem2+MSMT` for NCI-ISBI2013, please run `scripts/eccv_nci/Mem2_MSMT.sh`

**For each of the script, please ensure to modify the following before executing:**
- Replace all occurences of the `<USERNAME>` token with your specified value properly. You might need to create similar directory structures as specified in these scripts on your own system.
- Specify `device=?` in the script. If you have two gpu devices for example, `device` could be `0` or `1`. 

## Generate tables in paper
After the training finished, you can run `gen_paper.py` to generate the running average dice numbers in the tables. Please properly specify meta information in `paper_data.py`.

Note that you also need to replace all occurences of the `<USERNAME>` token with your specified value properly in these codes.

## Contact the author
Should there be any questions, please contact the author directly at the kevinmtian@gmail.com

