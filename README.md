# OpenTAL: Towards Open Set Temporal Action Localization
[Project](https://www.rit.edu/actionlab/opental) **|** [Paper & Supp](https://arxiv.org/pdf/2203.05114.pdf) **|** [Slides](assets/OpenTAL-Oral.pdf)

[Wentao Bao](https://cogito2012.github.io/homepage), 
[Qi Yu](https://www.rit.edu/mining/qi-yu), 
[Yu Kong](https://people.rit.edu/yukics/)

This is an official PyTorch implementation of OpenTAL, accepted as an Oral paper in IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR Oral**), 2022.


## Open Set Temporal Action Localization (**OSTAL**) Task

The OSTAL task is different from the TAL in that, there exist unknown actions in untrimmed videos and the OSTAL models need to reject the positively localized action as the unknown, rather than falsely assign it to a known class such as the `LongJump` or the non-informative `Background` class.

<p align="center">
<img src="assets/OSTAL.png" alt="OSTAL" width="600px"/>
</p>

## OpenTAL Method
In this work, we, for the first time, step toward the Open Set TAL (OSTAL) problem and propose a general framework **OpenTAL** based on Evidential Deep Learning (EDL). Specifically, the OpenTAL consists of uncertainty-aware action classification, actionness prediction, and temporal location regression. With the proposed importance-balanced EDL method, classification uncertainty is learned by collecting categorical evidence majorly from important samples. To distinguish the unknown actions from background video frames, the actionness is learned by the positive-unlabeled learning. The classification uncertainty is further calibrated by leveraging the guidance from the temporal localization quality. The OpenTAL is general to enable existing TAL models for open set scenarios, and experimental results on THUMOS14 and ActivityNet1.3 benchmarks show the effectiveness of our method. 

The following figure shows an overview of our proposed OpenTAL method. 

![opental](assets/opental.png)

## :boom: Updates
- (08-15-2023) If you need the preprocessed RGB data but do not have access to Weiyun links, please send to me with your email that I can share with you the OneDrive links. Thanks!
- (03-28-2022) OpenTAL paper is selected for an ORAL presentation! :tada: :tada: :tada:
- (03-11-2022) We released OpenTAL training and inference code, and open-set data splits.
- (03-02-2022) OpenTAL is accepted by CVPR 2022.

## Getting Started

This repo is developed mainly referring to the [AFSD (CVPR 2021)](https://github.com/TencentYoutuResearch/ActionDetection-AFSD), a great TAL work from Tencent YouTu Research. Most installations and data preparation steps are kept unchanged.
### Environment
- Python 3.7
- PyTorch == 1.9.0
- NVIDIA GPU

### Setup
```shell script
# create & activate a conda virtual environment
conda create -n opental python=3.7 
conda activate opental

# install pytorch and cudatoolkit (take pytorch 1.9 as an example)
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit-dev -c pytorch -c conda-forge

# install other python libs.
pip install -r requirements.txt

# compile the C++/CU files from AFSD
python setup.py develop
```


### Data Preparation
- **THUMOS14 RGB data:**
1. Download pre-processed RGB npy data (13.7GB): [\[Weiyun\]](https://share.weiyun.com/bP62lmHj)
2. Unzip the RGB npy data to `./datasets/thumos14/validation_npy/` and `./datasets/thumos14/test_npy/`
3. Download the annotations and our released **Open-Set Splits**: [\[THUMOS14 Annotations\]](https://drive.google.com/drive/folders/1dQUIhZYfmKoMLJSa79g2XHmvP_NuGtQ7?usp=sharing), and unzip them to `./datasets/thumos14/`

- **ActivityNet 1.3 RGB data:**
1. Download pre-processed videos (32.4GB): [\[Weiyun\]](https://share.weiyun.com/PXXtHcbp)
2. Run the AFSD data processing tool to generate RGB npy data: `python3 AFSD/anet_data/video2npy.py THREAD_NUM`
3. Download the annotations and our released **Open-Set Splits**: [\[ActivityNet1.3 Annotations\]](https://drive.google.com/drive/folders/163pxhHoSungM7cE0ZQu6_idGnW6y85wF?usp=sharing), and unzip them to `./datasets/activitynet/`

### Inference
We provide the pretrained models that contain I3D backbone model and OpenTAL final models on three open-set splits of THUMOS14:
[\[Google Drive\]](https://drive.google.com/drive/folders/1lEospHdatqUvKQi4ODSaLm07timdmYyV?usp=sharing)

```shell script
cd experiments/opental
# run OpenTAL model inference, using THUMOS14 splits as the unknown.
bash test_opental_final.sh 0  # GPU_ID=0

# run OpenTAL model inference, using ActivityNet as the unknown.
bash test_opental_cross_data.sh 0  # GPU_ID=0
```
Results will be saved in: `./output/opental_{final|crossdata}/split_{0|1|2}/thumos14_{open_rgb|anet_merged}.json`.


### Evaluation
The output json results and evaluation reports of our pretrained model can be downloaded from: [\[Google Drive\]](https://drive.google.com/drive/folders/1CxW9vkNTzo3mOk9BYbOgTfBkXJu6qn7S?usp=sharing)
```shell script
cd experiments/opental
# evaluate the inference results on dataset using THUMOS14 splits as the unknown.
bash eval_opental_final.sh

# evaluate the inference results on dataset using ActivityNet as the unknown.
bash eval_opental_cross_data.sh
```
Final results will be reported on your shell terminal, intermediate results on each dataset split are saved in `./output/opental_{final|crossdata}/split_{0|1|2}/`


### Training
```shell script
# train the OpenTAL model on a specific split (0/1/2), e.g., SPLIT=0
cd experiments/opental
SPLIT=0
GPU_ID=0
# with nohup command, the python process will be in backend for long-time running
nohup bash train_opental_final.sh ${GPU_ID} ${SPLIT} >train_opental.log 2>&1 &
```

To monitor your training status, you can either show the real-time text report or tensorboard visualization.
```shell script
# real-time text report
cd experiments/opental
tail -f train_opental.log

# tensorboard visualization (we only recorded the training for split_0)
cd models/thumos14/opental_final/split_0/tensorboard
tensorboard --logdir=./ --port=6789
```


## Citation
If you find this project useful for your research, please use the following BibTeX entries.
```
@InProceedings{Bao_2022_CVPR,
    author    = {Wentao Bao, Qi Yu, Yu Kong},
    title     = {OpenTAL: Towards Open Set Temporal Action Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```

```
@InProceedings{Lin_2021_CVPR,
    author    = {Lin, Chuming and Xu, Chengming and Luo, Donghao and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Fu, Yanwei},
    title     = {Learning Salient Boundary Feature for Anchor-free Temporal Action Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3320-3329}
}
```
