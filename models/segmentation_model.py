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

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu'):
        super(UNet3D, self).__init__()

        self.backbone = UNet3D()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, n_class) # 分割类别


    def load_params(self, model_dict):
        store_dict = self.backbone.state_dict()
        for key in model_dict.keys():
            if 'out_tr' not in key:
                store_dict[key.replace("module.", "")] = model_dict[key]
        self.backbone.load_state_dict(store_dict)
        print('Use pretrained weights')

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        # return self.out512

        return self.out512, self.out


class Unet_Cls_seg(nn.Module):
    def __init__(self, n_class=4) -> None:
        super(Unet_Cls_seg, self).__init__()

        self.encoder = UNet3D()
        self.da1 = DASModule(d_model=512, kernel_size=3, H=12, W=12)

        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
            )

        # self.feature_fusion = 1

        self.cls_head = nn.Sequential(
            # nn.Linear(in_features=512, out_features=128),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_class)
        )

    def load_params(self, model_dict):
        store_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(store_dict)
    #
    #     print('Use pretrained weights')
    def forward(self, x):
    # def forward(self, liver, spleen, left_kidney, right_kidney):
        cls_out, seg_out = self.UNet3D(x)

        cls_out = self.da1(cls_out)
        cls_out = self.GAP(cls_out)

        cls_pred = self.cls_head(cls_out)
        return cls_pred, seg_out


    if __name__ == "__main__":
        liver = torch.ones((2, 1, 96, 96, 96))
        spleen = torch.ones((2, 1, 96, 96, 96))
        left_kidney = torch.ones((2, 1, 96, 96, 96))
        right_kidney = torch.ones((2, 1, 96, 96, 96))
        model = UNet3D(n_class=4)
        # load pretrain model
        pretrain = "/home/zxg/zxg_code/TransUnet_Challenge/unet_classification/pretrain_model/unet.pth"
        model.load_params(torch.load(pretrain, map_location='cpu')['net'])
        # pred = model(liver, spleen, left_kidney, right_kidney)
        pred = model(liver)
        print(pred)