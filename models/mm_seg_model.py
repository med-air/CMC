import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.DANet import DAModule,DASModule

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class MIA_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(MIA_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X z*y*x)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out

class Encoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, act='relu'):
        super(Encoder, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        return self.out512,(self.skip_out64,self.skip_out128,self.skip_out256,self.skip_out512)

class Decoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu'):
        super(Decoder, self).__init__()


        # self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x,skips):

        self.out_up_256 = self.up_tr256(x,skips[2])
        self.out_up_128 = self.up_tr128(self.out_up_256,skips[1])
        self.out_up_64 = self.up_tr64(self.out_up_128,skips[0])
        self.out = self.out_tr(self.out_up_64)

        return self.out
class fusionLayer(nn.Module):
    def __init__(self, in_channel, outChans, depth,act):

        super(fusionLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = LUConv(1024, 512,act)
        self.layer2 = LUConv(512, 256,act)
    def forward(self, x1,x2):
        concat = torch.cat((x1,x2),1)
        cov_layer1 = self.layer1(concat)
        cov_layer2 = self.layer2(cov_layer1)
        out = self.sigmoid(cov_layer2)
        return out
class multimodal_segmentation(nn.Module):
    def __init__(self, n_class=4) -> None:
        super(multimodal_segmentation, self).__init__()
        self.conv3d = nn.Conv3d(2,1,kernel_size=1, stride=1,padding=0)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.MIA_module = MIA_Module(16)
        self.fusion_layer =fusionLayer(512,512, 1,act='relu')
        self.conv3d_convert = nn.Sequential(
            nn.GroupNorm(16, 768),
            nn.ReLU(inplace=True),
            nn.Conv3d(768,512, kernel_size=1, stride=1,padding=0)
        )

    def load_params(self, model_dict):
        encoder_store_dict = self.encoder.state_dict()
        decoder_store_dict = self.decoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                encoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
            if "up_tr" in key:
                decoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(encoder_store_dict)
        self.decoder.load_state_dict(decoder_store_dict)

        print('Use encoder pretrained weights')
    def forward(self, CT_img,MRI_img):
        CT_img_F_ds, CT_Skips = self.encoder(CT_img)
        MRI_img_F_ds, MRI_Skips = self.encoder(MRI_img)

        CT_img_F_mia = self.MIA_module(CT_img_F_ds)
        MRI_img_F_mia = self.MIA_module(MRI_img_F_ds)

        out_fuse = self.fusion_layer(CT_img_F_mia, MRI_img_F_mia)
        CT_F_z = torch.cat([out_fuse, CT_img_F_mia],dim=1)
        MRI_F_z = torch.cat([out_fuse, MRI_img_F_mia],dim=1)
        CT_F_z = self.conv3d_convert(CT_F_z)
        MRI_F_z = self.conv3d_convert(MRI_F_z)


        CT_seg_out = self.decoder(CT_F_z, CT_Skips)
        MRI_seg_out = self.decoder(MRI_F_z, MRI_Skips)

        return CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out
