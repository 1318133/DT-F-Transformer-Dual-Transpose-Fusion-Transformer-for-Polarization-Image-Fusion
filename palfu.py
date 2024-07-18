import torch.nn.functional as F
import torch.nn as nn
# from unet.unet_parts import *
import numpy
import torch
from .unet_model import UNet

from .unet_parts import *

from .liif import MLP, ifa_feat

import numbers
from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()

        self.con1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.con31 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.con32 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.con33 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.project_out = nn.Conv2d(dim*4, dim, kernel_size=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dw2 = maxpool(2)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dw4 = maxpool(4)

        # self.project_in = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.con1(x)
        x2 = self.con31(x)
        x3 = self.up2(self.con32(self.dw2(x)))
        x4 = self.up4(self.con33(self.dw4(x)))
        xcat = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.project_out(xcat)
        return x


##########################################################################

class Attention_fuse(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_fuse, self).__init__()
        proj_drop = 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.y1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.f1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.f1_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.y3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.f3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.f3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3)
        # self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x,y,f):
        b,c,h,w = x.shape

        x = self.x3(self.x1(x))
        y = self.y3(self.y1(y))
        f = self.f3(self.f1(f))
        f2 = self.f3_2(self.f1_2(f))
        
        x = rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        y = rearrange(y, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        f = rearrange(f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        f2 = rearrange(f2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)

        attn = (x @ y.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_a = (attn @ f)
        
        out_a = rearrange(out_a, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        attn2 = (y @x.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        out_b = (attn2 @ f2)
        
        out_b = rearrange(out_b, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = torch.cat((out_a,out_b),dim=1)

        out = self.project_out(out)#(outc)
        return out

############Fusion
class Dual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Dual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv2 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.project_out2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.block_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        # self.block_out = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        


    def forward(self, x,y):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        qkv2 = self.qkv_dwconv2(self.qkv2(y))
        q2,k2,v2 = qkv2.chunk(3, dim=1)   
        
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)


        attn = (q @ k2.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)+x


        attn2 = (q2 @ k.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = self.project_out2(out2)+y

        out_f = torch.cat((out,out2),dim=1)
        out_f = self.block_out(out_f)
        return out_f


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm12 = LayerNorm(dim, LayerNorm_type)
        self.norm13 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_fuse(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim)

    def forward(self, x,y,f):
        x = f + self.attn(self.norm1(x),self.norm12(y),self.norm13(f))
        x = x + self.ffn(self.norm2(x))

        return x



'''make for fusion'''
def fusion( en1, en2):
    f_0 = (en1 + en2)/2
    return f_0

def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    feat_grads = torch.abs(feat_grads)
    return feat_grads


def mask_fusion( x1, m1, x2, m2):
    mf = x1 * m1 +  x2 * m2
    return mf

def mask_addition( x1, x2):
    mf = x1 * 0.5 +  x2 * 0.5
    return mf

def Concat(x, y, z):
    return torch.cat((x, y), z)

class wtNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pthfile = 'F:/3line/checkpoints/CP_epoch13.pth'):
        super(wtNet, self).__init__()
         
        # net= UNet(n_channels=1, n_classes=1, bilinear=True)
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 
        dim = 48
        heads = 2
        LayerNorm_type = 'WithBias'

        # self.conv1 =  nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode='replicate')
        # self.conv2 =  nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode='replicate')

        self.tb1 = TransformerBlock(dim=dim, num_heads=heads, LayerNorm_type=LayerNorm_type)
        self.tb2 = TransformerBlock(dim=dim, num_heads=heads, LayerNorm_type=LayerNorm_type)
        self.tb3 = TransformerBlock(dim=dim, num_heads=heads, LayerNorm_type=LayerNorm_type)
        self.tb4 = TransformerBlock(dim=dim, num_heads=heads, LayerNorm_type=LayerNorm_type)


        self.conv1 =  nn.Conv2d(1, dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 =  nn.Conv2d(1, dim, kernel_size=3, padding=1, padding_mode='replicate')

        self.conv3 =  nn.Conv2d(2, dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv4 =  nn.Conv2d(dim, 1, kernel_size=3, padding=1, padding_mode='replicate')

        self.relu = nn.ReLU()
        self.tanh =nn.Tanh()
        self.unsher = nn.PixelUnshuffle(2)
        self.unsher = nn.PixelShuffle(2)


        if pthfile is not None:
            # self.load_state_dict(torch.save(torch.load(pthfile), pthfile,_use_new_zipfile_serialization=False), strict = False)  # 训练所有数据后，保存网络的参数
            self.load_state_dict(torch.load(pthfile), strict = False)
        


    def forward(self, xo, yo):
        x = self.conv1(xo)
        y = self.conv2(yo)
        fu = torch.cat((xo,yo),dim = 1)
        fu = self.relu(self.conv3(fu))
        fu1 = self.tb1(x,y,fu)
        fu2 = self.tb2(x,y,fu1)
        fu3 = self.tb3(x,y,fu2)
        fu4 = self.tb4(x,y,fu3)

        # fu1 = self.tb1(fu)
        # fu2 = self.tb2(fu1)
        # fu3 = self.tb3(fu2)
        # fu4 = self.tb4(fu3)

        fu = self.tanh(self.conv4(fu4))


        return fu,fu,fu#,x_mi,x_mi #x_mi,y_mi