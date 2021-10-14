import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import math
import sys
from torch.nn import functional as F
sys.dont_write_bytecode = True

''' 

	This Network is designed for Few-Shot Learning Problem. 
'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_MATANet(pretrained=False, model_root=None, which_model='MATA', norm='batch', init_type='normal',
                  use_gpu=True, **kwargs):
    MATANet = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'MATA':
        MATANet = MATA(norm_layer=norm_layer, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(MATANet, init_type=init_type)

    if use_gpu:
        MATANet.cuda()

    if pretrained:
        MATANet.load_state_dict(model_root)

    return MATANet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes: MATA_64F
##############################################################################

# Model: MATA_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Multi-scale feature generator --> AdaptiveTaskAttentionModule --> Calculate joint similarity
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class MATA(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, superparam_k=1):
        super(MATA, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.features1 = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

        )
        self.features2_4 = nn.Sequential(  # 3*84*84
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),  # 64*21*21
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21
        )

        self.ATAM = AdaptiveTaskAttentionModule(superparam_k=superparam_k)
        self.CBAM = Channel_Spatial_Attention(64)
    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.features1(input1)
        q = self.CBAM(q)
        q = self.features2_4(q)
        # extract features of input2--support set=
        Sup = []

        for i in range(len(input2)):
            support_set_sam = self.features1(input2[i])
            support_set_sam = self.CBAM(support_set_sam)
            support_set_sam = self.features2_4(support_set_sam)
            Sup.append(support_set_sam)

        x = self.ATAM(q, Sup)

        return x




###############################################################################
# Attention module: Channel & Spatial Attention
###############################################################################

class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel = 64, ratio = 16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class Channel_Spatial_Attention(nn.Module):
    def __init__(self, channel):
        super(Channel_Spatial_Attention, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# ========================== Define an image-to-class layer ==========================#

#generate multi-scale features
class MultiGrained_generator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(MultiGrained_generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.trans = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
        )
        self.pool_1 = nn.AvgPool2d(2)
        self.pool_2 = nn.AvgPool2d(3)
        self.pool_3 = nn.AvgPool2d(5)
        self.pool_4 = nn.AvgPool2d(7)

    def forward(self, x):

        x1 = self.trans(x)
        x2 = self.pool_1(x1)
        x3 = self.pool_2(x1)
        x4 = self.pool_3(x1)
        x5 = self.pool_4(x1)

        return x1,x2,x3,x4,x5


class AdaptiveTaskAttentionModule(nn.Module):
    def __init__(self, superparam_k=1):
        super(AdaptiveTaskAttentionModule, self).__init__()
        self.superparam_k = superparam_k
        self.MG = MultiGrained_generator()

    # Calculate the tasK attention score and select k most similar LRs
    def attention_and_select(self, input1, input2):
        B, C, h, w = input1.size()
        Similarity_list = []
        S = []
        for k in range(len(input2)):
            support_set_sam = input2[k]
            f1, f2, f3, f4, f5 = self.MG(support_set_sam)
            f1 = f1.permute(1, 0, 2, 3)
            f1 = f1.contiguous().view(C, -1)
            f2 = f2.permute(1, 0, 2, 3)
            f2 = f2.contiguous().view(C, -1)
            f3 = f3.permute(1, 0, 2, 3)
            f3 = f3.contiguous().view(C, -1)
            f4 = f4.permute(1, 0, 2, 3)
            f4 = f4.contiguous().view(C, -1)
            f5 = f5.permute(1, 0, 2, 3)
            f5 = f5.contiguous().view(C, -1)
            f = torch.cat((f1, f2, f3, f4, f5), dim=1)
            f_norm = torch.norm(f, 2, 0, True)
            f = f / f_norm
            S.append(f)
        global_support = torch.cat(S, 1)

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.unsqueeze(0)
            q1, q2, q3, q4, q5 = self.MG(query_sam)
            q1 = q1.contiguous().view(C, -1)
            q2 = q2.permute(1, 0, 2, 3)
            q2 = q2.contiguous().view(C, -1)
            q3 = q3.permute(1, 0, 2, 3)
            q3 = q3.contiguous().view(C, -1)
            q4 = q4.permute(1, 0, 2, 3)
            q4 = q4.contiguous().view(C, -1)
            q5 = q5.permute(1, 0, 2, 3)
            q5 = q5.contiguous().view(C, -1)
            query_sam = torch.cat((q1, q2, q3, q4, q5), dim=1)

            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(S)).cuda()

            for j in range(len(S)):
                support_set_sam = S[j]
                mask = query_sam @ global_support
                mask = torch.sum(mask, dim=1)
                mask = F.normalize(mask, p=2, dim=0)
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam @ support_set_sam
                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.superparam_k, 1)
                topk_value= torch.sum(topk_value ,dim=1)
                topk_value = topk_value * mask
                inner_sim[0, j] = torch.sum(topk_value)

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list

    def forward(self, x1, x2):

        Similarity_list = self.attention_and_select(x1, x2)
        return Similarity_list



