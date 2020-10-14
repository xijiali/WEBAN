# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNetFinetune(nn.Module):
    def __init__(self, b_name='efficientnet-b0', num_classes=10):
        super(EfficientNetFinetune, self).__init__()
        self.efficientnet=EfficientNet.from_pretrained(b_name)
        self.fc1=nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Dropout(0.7),
            nn.Linear(in_features=1000, out_features=512,bias=True),
            nn.BatchNorm1d(512),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.BatchNorm1d(128),
        )
        self.classifier=nn.Linear(in_features=128, out_features=num_classes, bias=True)

    def forward(self, input):
        embedding=self.efficientnet(input)
        embedding=self.fc1(embedding)
        embedding=self.relu_fn(embedding)
        embedding=self.fc2(embedding)
        embedding=self.relu_fn(embedding)
        logits=self.classifier(embedding)
        return logits

    def relu_fn(self,x):
        """ Swish activation function """

        return x * torch.sigmoid(x)


# model=EfficientNetFinetune()
# inp=torch.randn(2,3,256,256)
# out=model(inp)
# print('out size {}'.format(out.size()))


