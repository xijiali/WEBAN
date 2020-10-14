# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True,
                 num_bottleneck=0, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.num_bottleneck=num_bottleneck
        if num_bottleneck>0:
            add_block = []
            if linear:
                add_block += [nn.Linear(input_dim, num_bottleneck)]
            else:
                num_bottleneck = input_dim
            if bnorm:
                add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if droprate > 0:
                add_block += [nn.Dropout(p=droprate)]
            add_block = nn.Sequential(*add_block)
            add_block.apply(weights_init_kaiming)

        classifier = []
        if num_bottleneck>0:
            classifier += [nn.Linear(num_bottleneck, class_num)]
        else:
            classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        if num_bottleneck > 0:
            self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if self.num_bottleneck>0:
            x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class HyperNetwork_FC(nn.Module):
    def __init__(self,n_gen,num_classes):
        super().__init__()
        self.n_gen = n_gen
        self.num_classes = num_classes
        self.criterion_softmax = nn.Softmax(dim=1)
        self.fc = ClassBlock(self.num_classes,self.n_gen)

    def forward(self, input_x):
        weights = self.fc(input_x)
        weights = self.criterion_softmax(weights)
        print('weights size is {}'.format(weights.size()))
        print('weights are : {}'.format(weights))
        return weights


class HyperNetwork_FC_dynamic_input(nn.Module):
    def __init__(self,n_gen,num_classes):
        super().__init__()
        self.n_gen = n_gen
        self.num_classes = num_classes
        self.criterion_softmax = nn.Softmax(dim=1)
        self.fc=ClassBlock(self.num_classes*self.n_gen,self.n_gen)

    def forward(self, input_x):
        weights = self.fc(input_x)
        weights = self.criterion_softmax(weights)
        #print('weights size is {}'.format(weights.size()))
        #print('weights are : {}'.format(weights))
        return weights