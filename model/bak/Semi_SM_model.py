from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.models import MIA_Module, Encoder, Decoder, FusionLayer
import torchio as tio
from functools import partial

class Semi_SM_model(nn.Module):
    def __init__(self, img_size, n_class=16, backbone = 'VIT3D'):
        super().__init__()
        self.backbone_name = backbone
        # this backbone uses the pre-trained model of "SAM-med3D"
        if backbone == 'VIT3D':
            self.encoder_depth = 12
            self.encoder_embed_dim = 384
            self.image_size = img_size
            self.encoder_num_heads = 12
            # self.encoder_num_heads=6
            self.vit_patch_size = 2
            # self.vit_patch_size = 16
            self.image_embedding_size = self.image_size // self.vit_patch_size
            self.prompt_embed_dim = 384
            self.encoder_global_attn_indexes = [2, 5, 8, 11]
            self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            self.decoder = Decoder()
            self.MIA_module = MIA_Module(16)
            self.fusion_layer = FusionLayer(512, 512, 1, act='relu')
            self.conv3d_convert = nn.Sequential(
                # nn.GroupNorm(16, 768),
                nn.GroupNorm(16, 1024),
                nn.ReLU(inplace=True),
                # nn.Conv3d(768,512, kernel_size=1, stride=1,padding=0)
                nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0)
            )
        # this backbone uses the pre-trained model of "CLIP-Driven-Universal-Model"
        elif backbone == 'Foundation_model':
            self.image_encoder = Encoder()
            self.decoder = Decoder()
            self.MIA_module = MIA_Module(16)
            self.fusion_layer = FusionLayer(512, 512, 1, act='relu')
            self.conv3d_convert = nn.Sequential(
                nn.GroupNorm(16, 1024),
                nn.ReLU(inplace=True),
                nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

    def load_encoder_params(self, model_dict):
        encoder_store_dict = self.image_encoder.state_dict()
        if self.backbone_name == 'VIT3D':
            for key in model_dict.keys():
                if "image_encoder.block" in key:
                    encoder_store_dict[key] = model_dict[key]
                elif "image_encoder.patch_embed" in key:
                    encoder_store_dict[key] = model_dict[key]
                else:
                    continue
            self.image_encoder.load_state_dict(encoder_store_dict,strict=False)
            print('Use VIT3D of SAM-3D pretrained weights')
        elif self.backbone_name == 'Foundation_model':
            for key in model_dict.keys():
                if "down_tr" in key:
                    encoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
            self.image_encoder.load_state_dict(encoder_store_dict)
            print('Use Foundation_model pretrained weights')

    def load_decoder_params(self, model_dict):
        decoder_store_dict = self.decoder.state_dict()
        for key in model_dict.keys():
            if "up_tr" in key:
                decoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.decoder.load_state_dict(decoder_store_dict)
        print('Use pretrained weights')

    def forward(self, CT_img,MRI_img):
        if self.backbone_name=="VIT3D":
            CT_img = self.norm_transform(CT_img.squeeze(dim=1))  # (N, C, W, H, D)
            CT_img = CT_img.unsqueeze(dim=1)
        CT_img_F_ds, CT_Skips = self.image_encoder(CT_img)
        MRI_img_F_ds, MRI_Skips = self.image_encoder(MRI_img)

        CT_img_F_mia = self.MIA_module(CT_img_F_ds)
        MRI_img_F_mia = self.MIA_module(MRI_img_F_ds)

        out_fuse = self.fusion_layer(CT_img_F_mia, MRI_img_F_mia)
        CT_F_z = torch.cat([out_fuse, CT_img_F_mia], dim=1)
        MRI_F_z = torch.cat([out_fuse, MRI_img_F_mia], dim=1)
        CT_F_z = self.conv3d_convert(CT_F_z)
        MRI_F_z = self.conv3d_convert(MRI_F_z)

        CT_seg_out = self.decoder(CT_F_z, CT_Skips)
        MRI_seg_out = self.decoder(MRI_F_z, MRI_Skips)


        # CT_MRI_seg_out = torch.cat([CT_seg_out, MRI_seg_out], dim=1)
        return CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out
        # return CT_seg_out, MRI_seg_out
        # return CT_MRI_seg_out