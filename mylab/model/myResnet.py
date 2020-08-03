# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:20:10 2020

@author: gejianwen
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#实现的resnet种类和与训练参数的地址
__all__=['conv3x3','conv3x1','conv1x1',
         'BasicBlock','Bottleneck','BasicBlock3x1',
         'ResNet','resnet18','resnet34','resnet50','resnet101','resnet152',
         'ResNetStockZT']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes,out_planes,stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

def conv3x1(in_planes,out_planes,stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=(3,1),stride=stride,padding=(1,0),bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)
    
    
#block
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv3x1(inplanes,planes,stride = (stride,1))
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x1(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)
        #只有通道数翻倍的时候，空间分辨率才会缩小
        #也就是只有每个大模块的第一次卷积会使用stide=2
        
        out+=residual
        out=self.relu(out)

        return out
    

class BasicBlock3x3(nn.Module):
    expansion=1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock3x3,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)
        #只有通道数翻倍的时候，空间分辨率才会缩小
        #也就是只有每个大模块的第一次卷积会使用stide=2
        
        out+=residual
        out=self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1=conv1x1(inplanes,planes)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=conv3x3(planes,planes,stride)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=conv1x1(planes,planes*self.expansion)
        #一般1x1会将通道缩小4倍，在这里还原
        
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)

        out+=residual
        out=self.relu(out)

        return out
    
class ResNet(nn.Module):
    
    def __init__(self, block,layers,num_classes=1000):
        '''
        block:残差模块的选择，普通/瓶颈
        layer：每个模块重复的次数
        numclasses：fc层的输出个数，也就是分类的个数
        '''
        super(ResNet, self).__init__()
        
        self.inplanes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #所有深度的resnet第一步操作都是相同的
        #input：224x224 output：56x56  conv1和maxpool都进行了downsample
        #这里的第一步消耗掉的信息太多了，目的应该是为了节省内存空间
        
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        #每一层通道翻倍，空间分辨率宽高减半
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                nn.BatchNorm2d(planes*block.expansion),
                )
            #如果stride！=1，说明x也需要降采样。
            #但是这里空间降采样的同时重组了通道信息。

        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x

# 用三个basic block
class ResNetStockZT(nn.Module):
    
    def __init__(self, block,layers,num_classes=1000):
        '''
        block:残差模块的选择，普通/瓶颈
        layer：每个模块重复的次数
        numclasses：fc层的输出个数，也就是分类的个数
        '''
        super(ResNetStockZT, self).__init__()
        
        self.inplanes=64
        self.conv1=nn.Conv2d(1,64,kernel_size=(7,1),stride=(1,1),padding=(3,0),bias=False) # 64*18 ->64*18
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
#        self.maxpool=nn.MaxPool2d(kernel_size=(3,1),stride=1,padding=(1,0))  # 32*16
        #所有深度的resnet第一步操作都是相同的
        #input：224x224 output：56x56  conv1和maxpool都进行了downsample
        #这里的第一步消耗掉的信息太多了，目的应该是为了节省内存空间
        
        self.layer1=self._make_layer(block,64,layers[0]) # 64*18
        self.layer2=self._make_layer(block,128,layers[1],stride=2) # 32*18
        self.layer3=self._make_layer(block,256,layers[2],stride=2) # 16*18
        self.layer4=self._make_layer(block,512,layers[3],stride=2) # 8*18
        #每一层通道翻倍，空间分辨率宽高减半
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(256*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                nn.BatchNorm2d(planes*block.expansion),
                )
            #如果stride！=1，说明x也需要降采样。
            #但是这里空间降采样的同时重组了通道信息。

        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
#        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x


def resnet18(pretrained=False,**kwargs):
    model=ResNet(BasicBlock,[2,2,2,2],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False,**kwargs):
    model=ResNet(BasicBlock,[3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False,**kwargs):
    model=ResNet(Bottleneck,[3,4,6,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False,**kwargs):
    model=ResNet(Bottleneck,[3,4,23,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False,**kwargs):
    model=ResNet(Bottleneck,[3,8,36,3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
















