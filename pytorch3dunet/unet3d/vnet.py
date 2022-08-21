import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Implementation of this model is borrowed and modified
(to support multi-channels and latest pytorch version)
from here:
https://github.com/Dawn90/V-Net.pytorch
"""

def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
            # self.do1 = nn.Dropout3d(p=0.6)
            # self.do1 = nn.Dropout3d(p=0.4)
            # self.do1 = nn.Dropout3d(p=0.2)
            # self.do1 = nn.Dropout3d(p=0.0)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        # self.do2 = nn.Dropout3d(p=0.6)
        # self.do2 = nn.Dropout3d(p=0.4)
        # self.do2 = nn.Dropout3d(p=0.2)
        # self.do2 = nn.Dropout3d(p=0.0)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
            # self.do1 = nn.Dropout3d(p=0.6)
            # self.do1 = nn.Dropout3d(p=0.4)
            # self.do1 = nn.Dropout3d(p=0.2)
            # self.do1 = nn.Dropout3d(p=0.0)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        # import pdb; pdb.set_trace()
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        # import pdb; pdb.set_trace()
        # print(out.size())
        # print(skipxdo.size())

        if out.size() != skipxdo.size():
            out = F.pad(out, (skipxdo.size(4) - out.size(4),0,skipxdo.size(3) - out.size(3),0,skipxdo.size(2) - out.size(2),0), mode="replicate")

        # if out.size(2) < skipxdo.size(2):
        #   pd_sz = skipxdo.size(2) - out.size(2)
        #   out = F.pad(out, (0,0,0,0,pd_sz,0))
        # elif out.size(3) < skipxdo.size(3):
        #   pd_sz = skipxdo.size(3) - out.size(3)
        #   out = F.pad(out, (pd_sz,0,pd_sz,0,0,0))
        # print(out.size())
        # print(skipxdo.size())
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class AbstractVNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=4, final_sigmoid=True, is_segmentation=True, testing=False, is_conbr_head=True, is_kd_head=True):
        """
        is_conbr: is create parallel branch for contrastive loss
        is_kd: is create parallel branch for kd head        
        """
        super(AbstractVNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.is_conbr_head = is_conbr_head
        self.is_kd_head = is_kd_head

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        if self.is_conbr_head:
            # a parallel branch of self.down_tr128, pass to contrastive loss serving as multi-task regularizations
            self.down_tr128_conbr = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        # direct to two heads, 
        # the target head is going to predict the target
        # the kd head is to predict previous model's prediction on the new image, serving as multi-task regularization
        # distilling previous model's representations
        self.out_tr_target = OutputTransition(32, classes, elu)
        if self.is_kd_head:
            self.out_tr_kd = OutputTransition(32, classes, elu)

        self.testing = testing

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([4, 1, 30, 55, 55])
        out16 = self.in_tr(x)
        # (Pdb) out16.size()
        # torch.Size([4, 16, 30, 55, 55])
        out32 = self.down_tr32(out16)
        # (Pdb) out32.size()
        # torch.Size([4, 32, 15, 27, 27])
        out64 = self.down_tr64(out32)
        # (Pdb) out64.size()
        # torch.Size([4, 64, 7, 13, 13])

        out128 = self.down_tr128(out64)
        if self.is_conbr_head:
            # branched for multi-task fashion of contrastive regularizer
            out128_conbr = self.down_tr128_conbr(out64)
        else:
            out128_conbr = None

        # (Pdb) out128.size()
        # torch.Size([4, 128, 3, 6, 6])
        out256 = self.down_tr256(out128)
        # (Pdb) out256.size()
        # torch.Size([4, 256, 1, 3, 3])
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        # out_late is used for late contrastive learning
        out_late = self.up_tr64(out, out32)
        out = self.up_tr32(out_late, out16)

        head_target = self.out_tr_target(out)
        if self.is_kd_head:
            head_kd = self.out_tr_kd(out)
        else:
            head_kd = None

        if self.testing and self.final_activation is not None:
            head_target = self.final_activation(head_target)
            if self.is_kd_head:
                head_kd = self.final_activation(head_kd)
        # return out128 as semantic spatial features, used for contrastive loss
        # out32 is a higher resolution spatial feature, including more details
        
        # return order: prediction_head, kd_head, con_head, con_early_head, con_branch_head, [optional] con_late_head
        return head_target, head_kd, out128, out32, out128_conbr, out_late


# class VNetLight(nn.Module):
#     """
#     A lighter version of Vnet that skips down_tr256 and up_tr256 in oreder to reduce time and space complexity
#     """

#     def __init__(self, elu=True, in_channels=1, classes=4):
#         super(VNetLight, self).__init__()
#         self.classes = classes
#         self.in_channels = in_channels

#         self.in_tr = InputTransition(in_channels, elu)
#         self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
#         self.up_tr64 = UpTransition(128, 64, 1, elu)
#         self.up_tr32 = UpTransition(64, 32, 1, elu)
#         self.out_tr = OutputTransition(32, classes, elu)

#     def forward(self, x):
#         out16 = self.in_tr(x)
#         out32 = self.down_tr32(out16)
#         out64 = self.down_tr64(out32)
#         out128 = self.down_tr128(out64)
#         out = self.up_tr128(out128, out64)
#         out = self.up_tr64(out, out32)
#         out = self.up_tr32(out, out16)
#         out = self.out_tr(out)

#         

#         return out
