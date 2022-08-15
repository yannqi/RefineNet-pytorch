import torch.nn as nn
import torch.nn.functional as F



class ResidualConvUnit(nn.Module):
    """RCU: Residual Conv Unit."""
    def __init__(self, features):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):
    """Multi-resolution Fusion """
    def __init__(self, out_feats, *shapes): #变量前加*，则多余的函数参数会作为一个元组存在shapes中.
        super().__init__()

        _, max_h, max_w = max(shapes, key=lambda x: x[1]) #取第二位元素的最大值，TODO 可读性较差。

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, h, w = shape
            if max_h % h != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            self.scale_factors.append(max_h // h)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = self._un_pool(output, self.scale_factors[0])

        for i, x in enumerate(xs[1:], 1):
            tmp_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                tmp_out = self._un_pool(tmp_out, self.scale_factors[i])
            output = output + tmp_out

        return output
    def _un_pool(self, input, scale):
        return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)
    
    
    
class ChainedResidualPool(nn.Module):
    def __init__(self, feats, block_count=4):
        super().__init__()

        self.block_count = block_count
        self.relu = nn.ReLU(inplace=False)
        for i in range(0, block_count):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(0, self.block_count):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x

class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, *shapes) :
        super().__init__()    
        
        for i,shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats),residual_conv_unit(feats) 
                )
            )
        if len(shapes) != 1 :
            self.multi_resolution_fusion = multi_resolution_fusion(features, *shapes)
        else :
            self.multi_resolution_fusion = None
        
        self.chained_residual_pool = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)
    
    def forward(self, *xs):
        rcu_xs = []
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))  #走两个RCU

        if self.multi_resolution_fusion is not None:  #多输入的话，走一个MRF
            out = self.multi_resolution_fusion(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.chained_residual_pool(out) #走一个CRP
        return self.output_conv(out)  #最后再走一个RCU
    
    
class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)
        
        
#https://github.com/markshih91/refinenet_pytorch/blob/master/net/refinenet/blocks.py