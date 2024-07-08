# Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration

This is the PyTorch implementation of our MICCAI 2024 paper ["Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration"](https://github.com/med-air/CMC) by Xiaogen Zhou，Yiyou Sun, Min Deng, [Winnie Chiu Wing Chu*](https://scholar.google.com/citations?user=qgTwajMAAAAJ&hl=zh-CN&oi=ao)，and [Qi Dou*](https://www.cse.cuhk.edu.hk/~qdou/)
```
*denotes corresponding authors.
```

## Abstract
> Multimodal learning leverages complementary information derived from different modalities, thereby enhancing performance in medical image segmentation. However, prevailing multimodal learning methods heavily rely on extensive well-annotated data from various modalities to achieve accurate segmentation performance. This dependence often poses a challenge in clinical settings due to limited availability of such data. Moreover, the inherent anatomical misalignment between different imaging modalities further complicates the endeavor to enhance segmentation performance. To address this problem, we propose a novel semi-supervised multimodal segmentation framework that is robust to scarce labeled data and misaligned modalities. Our framework employs a novel cross modality collaboration strategy to distill modality-independent knowledge, which is inherently associated with each modality, and integrates this information into a unified fusion layer for feature amalgamation. With a channel-wise semantic consistency loss, our framework ensures alignment of modality-independent information from a feature-wise perspective across modalities, thereby fortifying it against misalignments in multimodal scenarios. Furthermore, our framework effectively integrates contrastive consistent learning to regulate anatomical structures, facilitating anatomical-wise prediction alignment on unlabeled data in semi-supervised segmentation tasks. Our method achieves competitive performance compared to other multimodal methods across three tasks: cardiac, abdominal multi-organ, and thyroid-associated orbitopathy segmentations. It also demonstrates outstanding robustness in scenarios involving scarce labeled data and misaligned modalities. 

## Usage
### Installation
#### 1. Download from GitHub
```
git clone https://github.com/med-air/CMC.git
cd CMC
```
#### 2. Create conda environment
```
conda create -n cmc python=3.8
pip install -r requirements.txt
conda activate cmc
```
### Dataset preparing
#### 1. Download the MS-CMRSeg dataset: https://zmiclab.github.io/zxh/0/mscmrseg19/
#### 2. Download the AMOS dataset: https://amos22.grand-challenge.org/
## Pre-trained Weights
You can directly download the pre-trained SAM-Med3D via this [link](https://drive.google.com/file/d/1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub/view) and put it under pretrain_model/.

### Data preprocessing
```
python monai_preprocessing_multi_modal.py
```
### Training
```
python main.py --model_name semi_sam3D_seg --rank 0 --distributed 0
```
### Testing
```
python inference.py --checkpoint_path './best_model/model_final.pth'
```
## Contact
If you have any questions, please feel free to leave issues here, or contact [xiaogenzhou@cuhk.edu.hk](xiaogenzhou@cuhk.edu.hk).
## Citation
```
@article{zhou2024cmc,
     title={Robust Semi-Supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration},
     author={Zhou, Xiaogen and Sun, Yiyou, and Deng, Min and Chu Winnie Chiu Wing and Dou, Qi},
     journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
     year={2024}
   }
```



