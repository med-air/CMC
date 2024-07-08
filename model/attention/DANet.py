import numpy as np
import torch
from torch import nn
from torch.nn import init
from model.attention.SelfAttention import ScaledDotProductAttention
from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class PositionAttentionModule(nn.Module):

    # def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
    #     super().__init__()
    def __init__(self,d_model,kernel_size,H,W):
        super(PositionAttentionModule,self).__init__()
        self.d_model = d_model
        self.H=H
        self.W=W
        self.kernel_size=kernel_size
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    # def __init__(self,d_model=512,kernel_size=3,H=7,W=7):

    def __init__(self,d_model,kernel_size,H,W):
        super(ChannelAttentionModule,self).__init__()
        self.d_model = d_model
        self.H=H
        self.W=W

        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=3,padding=(3-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y




class DAModule(nn.Module):
    def __init__(self, d_model, kernel_size,H,W):
        super(DAModule, self).__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.H = H
        self.W = W
        self.position_attention_module=PositionAttentionModule(d_model=d_model,kernel_size=kernel_size,H=H,W=W)
        self.channel_attention_module=ChannelAttentionModule(d_model=d_model,kernel_size=kernel_size,H=H,W=W)

    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DASModule(nn.Module):
    def __init__(self, d_model, kernel_size, H, W):
        super(DASModule, self).__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.H = H
        self.W = W
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.spatial_attention_module = SpatialAttention()

    def forward(self, input):
        bs, c, h, w, z = input.shape
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        s_out = self.spatial_attention_module(input)
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        c_out = c_out.view(bs, c, h, w)
        return (p_out + c_out)*s_out

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
    print(danet(input).shape)
