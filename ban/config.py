# # -*- coding: utf-8 -*-
# from ban.models.mlp import MLP
# from ban.models.lenet import LeNet
# from ban.models.resnet import ResNet18
# from ban.models.resnet import ResNet34
# from ban.models.resnet import ResNet50
# from ban.models.resnet import ResNet101
# from ban.models.resnet import ResNet152
# from ban.models.efficientnet import EfficientNetFinetune
#
# """
# add your model.
# from your_model_file import Model
# model = Model()
# """
#
# __factory = {
#     'resnet18':ResNet18(),
#     'efficientnetb0':EfficientNetFinetune()
# }
#
# # model = ResNet50()
# def get_model(name='resnet18'):
#     if name not in __factory:
#         raise KeyError("Unknown model:", name)
#     model = __factory[name]
#     return model
# -*- coding: utf-8 -*-
from ban.models.mlp import MLP
from ban.models.lenet import LeNet
from ban.models.resnet import ResNet18
from ban.models.resnet import ResNet34
from ban.models.resnet import ResNet50
from ban.models.resnet import ResNet101
from ban.models.resnet import ResNet152
from ban.models.cifar_style_resnet import resnet32

"""
add your model.
from your_model_file import Model
model = Model()
"""

# model = ResNet50()
def get_model(model_name,num_classes):
    if model_name=='resnet18':
        model = ResNet18(num_classes)
    elif model_name=='resnet34':
        model=ResNet34(num_classes)
    elif model_name=='resnet32':
        model=resnet32(num_classes)
    else:
        print('Wrong model.')
    return model
