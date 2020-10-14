# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ban import config


class BANUpdater(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name=kwargs['model_name']
        self.last_model = None
        self.gen = 0

    def update(self, inputs, targets, criterion):
        self.optimizer.module.zero_grad()
        outputs = self.model(inputs)
        if self.gen > 0:
            teacher_outputs = self.last_model(inputs).detach()
            ce_loss,kld_loss = self.kd_loss(outputs, targets, teacher_outputs)
            loss=ce_loss+kld_loss
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        self.optimizer.module.step()
        if self.gen == 0:
            return loss
        else:
            return ce_loss, kld_loss

    def register_last_model(self, weight,device_ids,num_classes):
        self.last_model = config.get_model(self.model_name,num_classes).cuda()
        self.last_model.load_state_dict(torch.load(weight))
        self.last_model=nn.DataParallel(self.last_model, device_ids=device_ids)

    def kd_loss(self, outputs, labels, teacher_outputs, alpha=0.2, T=20):
        kld_loss=nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * alpha
        ce_loss=F.cross_entropy(outputs, labels) * (1. - alpha)
        return ce_loss,kld_loss

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen


class BANUpdater_ensemble(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.last_model = None
        self.gen = 0

    def update(self, inputs, targets, criterion):
        self.optimizer.module.zero_grad()
        outputs = self.model(inputs)
        tensor_size=outputs.size()#[64,10]
        if self.gen > 0:
            ensemble_teacher_outputs=torch.zeros(tensor_size).cuda()
            model_lst=self.last_model
            for i in range (len(model_lst)):
                ensemble_teacher_outputs+=model_lst[i](inputs).detach() # not change the saved state_dicts
            teacher_outputs = ensemble_teacher_outputs/len(model_lst)
            ce_loss,kld_loss = self.kd_loss(outputs, targets, teacher_outputs)
            loss=ce_loss+kld_loss
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        self.optimizer.module.step()
        if self.gen==0:
            return loss
        else:
            return ce_loss,kld_loss

    def register_last_model(self, weight,device_ids,num_classes):
        model_lst=[]
        for i in range (len(weight)):
            model=config.get_model(self.model_name,num_classes).cuda()
            model.load_state_dict(torch.load(weight[i]))
            model = nn.DataParallel(model, device_ids=device_ids)
            model_lst.append(model)
        self.last_model=model_lst


    def kd_loss(self, outputs, labels, teacher_outputs, alpha=0.2, T=20):
        kld_loss=nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * alpha
        ce_loss=F.cross_entropy(outputs, labels) * (1. - alpha)
        return ce_loss,kld_loss

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen


class BANUpdater_ensemble_hypernetwork(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.hypernetwork = kwargs.pop("hypernetwork")
        self.optimizer = kwargs.pop("optimizer")
        self.hypernetwork_optimizer = kwargs.pop("hypernetwork_optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.last_model = None
        self.gen = 0

    def update(self, inputs, targets, criterion):
        self.optimizer.module.zero_grad()
        self.hypernetwork_optimizer.module.zero_grad()
        outputs = self.model(inputs)
        tensor_size=outputs.size()#[64,10]
        if self.gen > 0:
            # hypernetwork
            weights = self.hypernetwork(outputs)  # [64,2]
            #print('weights is : {}'.format(weights))
            ensemble_teacher_outputs=torch.zeros(tensor_size).cuda()
            model_lst=self.last_model
            for i in range (len(model_lst)):
                temp_weight=weights[:,i].unsqueeze(1).expand(tensor_size)
                previous_logits=model_lst[i](inputs).detach()
                ensemble_teacher_outputs+=temp_weight*previous_logits # not change the saved state_dicts
            teacher_outputs = ensemble_teacher_outputs
            ce_loss,kld_loss,ce_loss_hypernetwork = self.kd_loss(outputs, targets, teacher_outputs)
            loss=ce_loss+kld_loss+ce_loss_hypernetwork
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        self.optimizer.module.step()
        self.hypernetwork_optimizer.module.step()
        if self.gen==0:
            return loss
        else:
            return ce_loss,kld_loss,ce_loss_hypernetwork

    def register_last_model(self, weight,device_ids,num_classes):
        model_lst=[]
        for i in range (len(weight)):
            model=config.get_model(self.model_name,num_classes).cuda()
            model.load_state_dict(torch.load(weight[i]))
            model = nn.DataParallel(model, device_ids=device_ids)
            model_lst.append(model)
        self.last_model=model_lst


    def kd_loss(self, outputs, labels, teacher_outputs, alpha=0.2, T=20):
        kld_loss=nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * alpha
        ce_loss=F.cross_entropy(outputs, labels) * (1. - alpha)
        ce_loss_hypernetwork=F.cross_entropy(teacher_outputs, labels)* (1. - alpha)
        return ce_loss,kld_loss,ce_loss_hypernetwork

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen



class BANUpdater_ensemble_hypernetwork_dynamic_input(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.hypernetwork = kwargs.pop("hypernetwork")
        self.optimizer = kwargs.pop("optimizer")
        self.hypernetwork_optimizer = kwargs.pop("hypernetwork_optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.last_model = None
        self.gen = 0

    def update(self, inputs, targets, criterion):
        self.optimizer.module.zero_grad()
        self.hypernetwork_optimizer.module.zero_grad()
        outputs = self.model(inputs)
        tensor_size=outputs.size()#[64,10]
        if self.gen > 0:
            model_lst=self.last_model
            ensemble_teacher_outputs=torch.zeros(tensor_size).cuda()
            concat_teacher_outputs=torch.Tensor().cuda()#[b,20]
            print('len model_lst is {}'.format(len(model_lst)))
            for i in range(len(model_lst)):
                previous_logits = model_lst[i](inputs).detach()
                concat_teacher_outputs=torch.cat((concat_teacher_outputs,previous_logits),dim=1)
            # hypernetwork
            weights = self.hypernetwork(concat_teacher_outputs)  # [64,2]
            # print('weights is : {}'.format(weights))
            for i in range (len(model_lst)):
                temp_weight=weights[:,i].unsqueeze(1).expand(tensor_size)
                previous_logits=model_lst[i](inputs).detach()
                ensemble_teacher_outputs+=temp_weight*previous_logits # not change the saved state_dicts
            teacher_outputs = ensemble_teacher_outputs
            ce_loss,kld_loss,ce_loss_hypernetwork = self.kd_loss(outputs, targets, teacher_outputs)
            loss=ce_loss+kld_loss+ce_loss_hypernetwork
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        self.optimizer.module.step()
        self.hypernetwork_optimizer.module.step()
        if self.gen==0:
            return loss
        else:
            return ce_loss,kld_loss,ce_loss_hypernetwork

    def register_last_model(self, weight,device_ids,num_classes):
        model_lst=[]
        for i in range (len(weight)):
            model=config.get_model(self.model_name,num_classes).cuda()
            model.load_state_dict(torch.load(weight[i]))
            model = nn.DataParallel(model, device_ids=device_ids)
            model_lst.append(model)
        self.last_model=model_lst


    def kd_loss(self, outputs, labels, teacher_outputs, alpha=0.2, T=20):
        kld_loss=nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * alpha
        ce_loss=F.cross_entropy(outputs, labels) * (1. - alpha)
        ce_loss_hypernetwork=F.cross_entropy(teacher_outputs, labels)* (1. - alpha)
        return ce_loss,kld_loss,ce_loss_hypernetwork

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen
