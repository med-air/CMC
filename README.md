# Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration

This is the PyTorch implementation of our MICCAI 2024 paper ["Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration"]() by Xiaogen Zhon, Yiyou Sun, Min Deng,
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
   
   ```

## Datasets
>Note: You can download our datasets as follows, and put them into the folder 'dataset':
### 1. MS-CMRSeg 2019 dataset: [here](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html)
### 2. AMOS Dataset: [here](https://zenodo.org/records/7262581)

# Running Experiments
#### Pre-train
Our encoder and decoder use a Foundation model's [[link](https://github.com/ljwztc/CLIP-Driven-Universal-Model)] pre-trained weights [[link](https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth)] and pre-trained weights [[link](https://huggingface.co/ybelkada/segment-anything/blob/main/checkpoints/sam_vit_b_01ec64.pth)] in SAM-Med3D[[link](https://github.com/uni-medical/SAM-Med3D)]. You also can download them from [here](https://gocuhk-my.sharepoint.com/:f:/r/personal/xiaogenzhou_cuhk_edu_hk/Documents/CMC/pre-trained_weights?csf=1&web=1&e=tgVEMp) Please download them and put them into the folder 'pretrain_model' before running the following script.



```bash
#### Training stage

python main.py --backbone 'Foundation_model' --batch_size 4 --img_size 96

#### Testing stage
python test.py --backbone 'Foundation_model' 

```


#### Checkpoints

We also provide our model checkpoints for the experiments on the AMOS dataset as listed below (Mean Dice is the evaluation metric).

|     Training      |                      CT  (Mean Dice(%))                      |                     MRI (Mean Dice(%))                    | Checkpoint |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |:-----:|
|10% Labeled data |                            76.28                             |                            84.27                            |[[checkpoint]](https://drive.google.com/file/d/1e-P7TEOIDJ04edFy1Eix8bTl5ZRD3l-g/view?usp=sharing](https://drive.google.com/file/d/1e-P7TEOIDJ04edFy1Eix8bTl5ZRD3l-g/view?usp=sharing)) |
|20% Labeled data  | 84.57  | 89.05 | [[checkpoint]](https://drive.google.com/file/d/1wq60hlEPFhotwPM5tCxcFK-hjPBZ842L/view?usp=sharing](https://gocuhk-my.sharepoint.com/:u:/r/personal/xiaogenzhou_cuhk_edu_hk/Documents/CMC/checkpoint/model_20_perc_labeled.pt?csf=1&web=1&e=iLxpaB))|

>Note: Please download these checkpoints and put them into the folder 'checkpoint', then run the following script for testing to reproduce our experimental results.

```bash
python test.py --backbone 'Foundation_model'
```
## Citation
If this repository is useful for your research, please cite:
```
@article{2024cmc,
     title={Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration},
     author={Xiaogen Zhon, Yiyou Sun, Min Deng,
Winnie Chiu Wing Chu and Qi Dou},
     journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
     year={2024}
   }
```  
### Contact
If you have any questions, please feel free to leave issues here, or contact ‘xiaogenzhou@126.com’
