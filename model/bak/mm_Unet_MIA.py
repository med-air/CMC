import torch
import torch.nn as nn
import torch.nn.functional as F


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

class fusionLayer(nn.Module):
    def __init__(self, in_channel, outChans, depth, act):
        super(fusionLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        m_batchsize, C, depth, height, width = x1.size()
        fusion = self.sigmoid((x1+x2))
        # proj_value = x1.view(m_batchsize, C, -1)
        # out = torch.bmm(attention, proj_value)
        out = fusion.view(m_batchsize, C, depth, height, width)
        return out
class SCSELayer(nn.Module):
    def __init__(self, channel=32, reduction=8):
        super(SCSELayer, self).__init__()
        self.cse_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.sse_conv = nn.Conv3d(channel, 1, 1, padding=0)

    def forward(self, x):
        b, c, z, w, h = x.size()
        cse_y = self.cse_avg_pool(x).view(b, c)
        cse_y = self.cse_fc(cse_y).view(b, c, 1, 1, 1)
        sse_y = self.sse_conv(x)

        return x * cse_y.expand_as(x) + x * sse_y.expand_as(x)

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

# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out

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

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=14, act='relu'):
        super(UNet3D, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, n_class)

        self.MIA_module = MIA_Module(16)
        self.fusion_layer =fusionLayer(512,512, 1,act='relu')

    def forward(self, x1):## x1, x2 are two different modality data, such as CT and MRI
        ## encode modality data from x1
        self.out64_x1, self.skip_out64_x1 = self.down_tr64(x1)
        self.out128_x1,self.skip_out128_x1 = self.down_tr128(self.out64_x1)
        self.out256_x1,self.skip_out256_x1 = self.down_tr256(self.out128_x1)
        self.out512_x1,self.skip_out512_x1 = self.down_tr512(self.out256_x1)
        x1_img_F_ds = self.out512_x1
        MIA_feat_x1 = self.MIA_module(x1_img_F_ds)

        ## encode modality data from x2
        # self.out64_x2, self.skip_out64_x2 = self.down_tr64(x2)
        # self.out128_x2,self.skip_out128_x2 = self.down_tr128(self.out64_x2)
        # self.out256_x2,self.skip_out256_x2 = self.down_tr256(self.out128_x2)
        # self.out512_x2,self.skip_out512_x2 = self.down_tr512(self.out256_x2)
        # x2_img_F_ds = self.out512_x2
        # MIA_feat_x2 = self.MIA_module(x2_img_F_ds)
        # fuse_feat = self.fusion_layer(MIA_feat_x1, x1_img_F_ds)
        # MIA_x1 = fuse_feat + MIA_feat_x1
        # MIA_x2 = fuse_feat + MIA_feat_x2

        ## decode for modality data from x1
        self.out_up_256_x1 = self.up_tr256(MIA_feat_x1,self.skip_out256_x1)
        self.out_up_128_x1 = self.up_tr128(self.out_up_256_x1, self.skip_out128_x1)
        self.out_up_64_x1 = self.up_tr64(self.out_up_128_x1, self.skip_out64_x1)
        self.out_x1 = self.out_tr(self.out_up_64_x1)

        ## decode for modality data from x2
        # self.out_up_256_x2 = self.up_tr256(MIA_x2,self.skip_out256_x2)
        # self.out_up_128_x2 = self.up_tr128(self.out_up_256_x2, self.skip_out128_x2)
        # self.out_up_64_x2 = self.up_tr64(self.out_up_128_x2, self.skip_out64_x2)
        # self.out_x2 = self.out_tr(self.out_up_64_x2)

        # return x1_img_F_ds, x2_img_F_ds, self.out_x1, self.out_x2,
        return x1_img_F_ds, self.out_x1

