# Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration

This is the PyTorch implementation of our MICCAI 2024 paper ["Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration"]() by Xiaogen Zhou, Yiyou Sun, Min Deng,
Winnie Chiu Wing Chu and [Qi Dou\*](https://www.cse.cuhk.edu.hk/~qdou/).

\* denotes corresponding authors.

## Abstract

> Multimodal learning leverages complementary information
derived from different modalities, thereby enhancing performance in med
ical image segmentation. However, prevailing multimodal learning meth
ods heavily rely on extensive well-annotated data from various modal
ities to achieve accurate segmentation performance. This dependence
often poses a challenge in clinical settings due to limited availability
of such data. Moreover, the inherent anatomical misalignment between
different imaging modalities further complicates the endeavor to en
hance segmentation performance. To address this problem, we propose
a novel semi-supervised multimodal segmentation framework that is ro
bust to scarce labeled data and misaligned modalities. Our framework
employs a novel cross modality collaboration strategy to distill modality
independent knowledge, which is inherently associated with each modal
ity, and integrates this information into a unified fusion layer for fea
ture amalgamation. With a channel-wise semantic consistency loss, our
framework ensures alignment of modality-independent information from
a feature-wise perspective across modalities, thereby fortifying it against
misalignments in multimodal scenarios. Furthermore, our framework ef
fectively integrates contrastive consistent learning to regulate anatomi
cal structures, facilitating anatomical-wise prediction alignment on unla
beled data in semi-supervised segmentation tasks. Our method achieves
competitive performance compared to other multimodal methods across
three tasks: cardiac, abdominal multi-organ, and thyroid-associated or
bitopathy segmentations. It also demonstrates outstanding robustness in
scenarios involving scarce labeled data and misaligned modalities.

![](./Figures/my_flowchartl.png)

## Getting Started

#### Installation

1. Download from GitHub

   ```bash
   git clone https://github.com/med-air/CMC.git
   
   cd CMC
   ```

2. Create conda environment

   ```bash
   conda create --name CMC python=3.8.18
   conda activate CMC
   pip install -r requirements.txt
   # CUDA 11.8 
   conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   
   ```

## Datasets
>Note: You can download our datasets as follows, please download our pre-processing dataset of AMOS from [here](https://pan.quark.cn/list#/list/all/44b4447c7dea48c697cf9bdc70de1b35-CMC_data/02961a722377466d848782dd86d97fab-AMOS) and put them into the folder 'dataset/mm_amos/':
### 1. MS-CMRSeg 2019 dataset: [here](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html)
### 2. AMOS Dataset: [here](https://zenodo.org/records/7262581)

# Running Experiments
#### Pre-train
Our encoder and decoder use a Foundation model's [[link](https://github.com/ljwztc/CLIP-Driven-Universal-Model)] pre-trained weights [[link]([https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth](https://pan.quark.cn/s/2ce82bc2c684))]. Please download them and put them into the folder 'pretrained_model' before running the following script.



```bash
#### Training stage


## multi GPU for training with DDP 
python main.py --distributed

## single GPU for training 
python main.py

#### Testing stage
python test.py

```


#### Checkpoints

We also provide our model checkpoints for the experiments on the AMOS dataset as listed below (Mean Dice is the evaluation metric).

| Training  | Spleen Mean Dice(%)  |   Right kidney Mean Dice(%)     |  Left kidney Mean Dice(%)       |  Liver Mean Dice(%)  | Overall Mean Dice (%)  | Checkpoint |
| :-------: | :-------: | :-------: |:-------: |:-------: | :-------: |:-----:|
|CT modality: 10% (20%) Labeled data |       72.36 (77.80)       |          78.04 (81.78)  |    78.36 (79.07)  | 85.10 (87.55) | 78.46 (81.55)        |[[checkpoint]](https://pan.quark.cn/list#/list/all/44b4447c7dea48c697cf9bdc70de1b35-CMC_data/513be8de50ed4f7e850ee2b7816f1e24-pretrained_checkpoints/c05af5f325c944699cfd61a9ce0f4e61-saved_checkpoint)) |
|MRI modality: 10% (20%) Labeled data |       79.86 (80.21)      |          81.94 (85.45)  |    80.64 (87.54) | 86.10 (88.24)  | 81.70 (87.42)       |[[checkpoint]](https://pan.quark.cn/list#/list/all/44b4447c7dea48c697cf9bdc70de1b35-CMC_data/513be8de50ed4f7e850ee2b7816f1e24-pretrained_checkpoints/c05af5f325c944699cfd61a9ce0f4e61-saved_checkpoint)) |

>Note: Please download these checkpoints and put them into the folder './run/saved_checkpoint', then run the following script for testing to reproduce our experimental results.


## Citation
If this repository is useful for your research, please cite:
```
@article{2024cmc,
     title={Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration},
     author={Xiaogen Zhou, Yiyou Sun, Min Deng,
Winnie Chiu Wing Chu and Qi Dou},
     journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
     year={2024}
   }
```  
### Contact
If you have any questions, please feel free to leave issues here, or contact ‘xiaogenzhou@126.com’
