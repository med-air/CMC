# 代�| ~A说�~X~N
�~H~]�~[�~Z~D�~S�~^~\�~M�~A�~B�~N��~P~F�~@~_度�~L�~[| 此�~H~F�~I�模�~^~K�~H~Q们�~T��~Z~D�~X�nnUnet�~@~B�~E��~S�~Z~D训�~C�~N��~P~FnnUnet�~Z~D代�| ~A�~X�train.sh�~Z~D22-45�~L�~L�~B�~^~\�~G��~W��~X�~O��
以注�~G~J�~N~I�~@~B
nnUnet�~Z~D�~M�~N��~\~I�~B��~[��~Z��~H主�~A�~X��~P~D�~M步骤�~T�~C�~A�~P~P�~L�~[��~H�说5 fold训�~C�~R~L�~N��~K�~F�~L�~[~F�~P~H�~\��~@起�~]~^常麻�~C��~I�~L并�~T�~_�~W�~X��~]~^常常�~T��~Z~Dbb
aseline�~L�~O��~V�~H~F�~I�模�~^~K�~H~F�~U��~M| �~T�~M�~X�~H25%�~I�~L�~L�~\~[�~M�~N��~@~A�~H�~C��~_�~E解�~L�~H~Q们�~F�~H~F�~I��~S�~^~\�~[��~N��~T��~\�代�| ~A�~S中�~@~B
�~B�~^~\�~\~I�~\~@�~A�~M�~N��~H~F�~I�模�~^~K�~L�~O��~C��~\~@�~A�~[��~Z�~W��~W��~L�~L�~\~[�~@~A�~H�~C��~A~T系�~H~Q们�~L�~H~Q们�~]�~A�~H~F�~I��~S�~^~\�~X��~O��~M�~N��~Z~D�~@~B�~\~I任�~U�~W��~X请�~~
A~T系mrjiang@cse.cuhk.edu.hk

�~H~Q�~[�信�~\�次�~T�~[�~Z~D�~G~M�~B��~X��~H~F类模�~^~K�~L�~[| 此�~\�代�| ~A主�~A�~X��~M�~N��~H~F类模�~^~K�~H75%�~I�~@~B

## 误差
�~\~K�~H~F�~I�模�~^~K�~Z~D�~C~E�~F��~L�~H~Q们�~G~G�~T�nnUnet�~H~F�~I��~Z~D�~L�~U�~_�~X��~M~C�~T��~Y��~B��~Z~Dbaseline�~L�~H~F�~I�误差�~T该�~M�~X��~I��~H�大�~@~B
�~H~F类模�~^~K�~Z~D�~C~E�~F��~L误差�~T该�~X好�~L�~H~F�~U�大�~B2-3�~H~F�~L�~O��~C��~X�~O��~C��~N�~_

## �~D�~K�~@��~C�
nnUnet�~Z~D训�~C�~W好�~Z天�~L�~N��~P~F�~_�~W�~M~A�~G| 个�~O�~W��~[
�~H~F类模�~^~K�~Z~D训�~C差�~M�~Z1-2天�~L�~N��~P~F差�~M�~Z1�~H~F�~R~_�~@~B

## �~N��~C�~E~M置�~H�~E�~@~I�~I
�~H~Q们�~G~G�~T��~Z~D�~X��~N��~Cpython==3.8.0, pytorch==1.13, 对�~T�~Z~Ddtk�~X�23.04
请�~\��~G~L�~]��~I�~E对�~T�~Z~Dpytorch�~R~Ltorchvision�~I~H�~\�
�~O��~V�~H~Q们主�~A�~T��~H��~Z~D�~L~E�~X�numpy, scipy, monai, pandas �~B�~^~\�~I�~E�~M对请�~@~A�~H�~G��~L�~I�~E�~@~B

module purge # �~J| 载�~I~M使�~T�module purge�~E�~P~F�~N��~C
module load anaconda3/5.2.0
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/gcc-7.3.1
module load compiler/dtk/23.04

conda create -n torch113_dtk2304 python==3.8
source activate torch113_dtk2304
## �~I�~Epytorch �~R~L torchvision
pip install -r requirements.txt

## �~D训�~C模�~^~K�~H�~E�~@~I�~I
使�~T��~FCLIP-Driven-Universal-Model�~Z~D�~D训�~C模�~^~K�~L�~K载�~S��~N��~Z
https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth
项�~[�Github�~\��~]~@�~Zhttps://github.com/ljwztc/CLIP-Driven-Universal-Model


## �~W�~U�~F�~J~B�~H�~E�~@~I�~I
模�~^~K�~T��~Z~D�~X�Unet�~Z~Dencoder�~L�~H~F类头�~T��~Z~D�~X�两�~Bmlp.
1. learning rate: 1e-4
2. learning decay: 1e-5
3. batch size: 2
4. epoch: 400
5. optimizer: AdamW
6. scheduler: LinearWarmupCosineAnnealingLR
7. loss_function: BCEWithLogitsLoss

### �~U��~S�~@~]路�~K�~M�~H�~E�~@~I�~I
�~H~Q们�~H��~T�segmentation map�~H��~O~V�~G�4个organ�~Z~D�~L��~_~_ (96*96*96)�~L�~D��~P~N�~H��~T��~D训�~C好�~Z~Dencoder�~C��~H~F(�~H~Q们�~Y�~G~L�~T��~Z~D�~X�CLIP-Driven Unet�~Z~Dencoder)�~F�~Y�~[organ�~O~~
X�~H~Pfeature�~Lconcate�~Y�~[feature�~D��~P~N�~[�~L�~H~F类�~@~B


### �~V��~U�~Z~D�~H~[�~V��~B��~H�~B�~^~\�~\~I�~I

## 训�~C�~A�~K�~H�~E�~@~I�~I
�~V�~E~H对�~U��~M��~[�~L�~D�~D�~P~F(monai_preprocessing.py), �~D��~P~N�~F�~U��~M��~L~I�~E�segmentation map�~[�~Lcrop(crop_to_size.py)�~W�~H�4个organ�~Z~D�~U��~M��~@~B�~D��~P~N�~[�~L训�~C(sbatch train.ss
h), �~I�~H��~E��~C��~L�~H~P训�~C�~P~N�~F~M�~[�~L�~N��~P~F(sbatch test.sh)

## �~E��~V注�~D~O�~K项
训�~C�~I~M�~Z~D�~U��~M��~D�~D�~P~F�~O��~C��~Z�~T�~C�~E��~L�~J�费�~@个�~Z�~O�~W��~C��~X�正常�~Z~D�~L�~O�以�~\�model/tmp_data/�~K�~]��~\~K�~H��~D�~D�~P~F�~P~N�~]�~X�~Z~D�~U��~M��~@~B
�~M�~N��~G�~K�~\~I任�~U�~W��~X请�~A~T系mrjiang@cse.cuhk.edu.hk, �~H~Q们�~Z确�~]�~C��~_�~M�~N��~@~B
