# -*- coding: utf-8 -*-
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
#added
from tensorboardX import SummaryWriter

from ban import config
from ban.updater import BANUpdater_ensemble
from common.logger import Logger
#added
import numpy as np
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_gen", type=int, default=10)
    parser.add_argument("--resume_gen", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--outdir", type=str, default="WEBAN_baseline_hypernetwork_input_logits_run1")
    parser.add_argument("--print_interval", type=int, default=50)
    #added
    parser.add_argument("--single_test", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default='model0.pth.tar')
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--root_dir", type=str, default='/gruntdata4/xiaoxi.xjl/classification_datasets')
    parser.add_argument("--model_name", type=str, default="resnet32")

    args = parser.parse_args()

    setup_seed(96)

    logger = Logger(args)
    logger.print_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    # added
    device_ids=[0,1]

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root=args.root_dir,
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root=args.root_dir,
                          train=False,
                          download=True,
                          transform=transform)
    else:
        trainset = MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=transform)
        testset = MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = config.get_model().to(device)
    model=nn.DataParallel(model,device_ids=device_ids)

    if args.evaluate: # not consider train from pretrain weights
        model_lst = []
        for i in range(args.n_gen):
            temp_model = config.get_model().to(device)
            if args.single_test:
                model_name = args.model_name
            else:
                model_name='model'+str(i)+'.pth.tar'
            temp_model.load_state_dict(torch.load(os.path.join(args.weight,model_name)))
            # print('{}-th model'.format(i))
            # for k, v in temp_model.named_parameters():
            #     if k == 'bn1.weight':
            #         print('k {}'.format(k))
            #         print('v {}'.format(v))
            model_lst.append(temp_model)

        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs=torch.zeros(inputs.size(0),args.n_classes).to(device)
                for j in range(args.n_gen):
                    outputs+=model_lst[j](inputs)
                outputs=outputs/args.n_gen
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().float()
        acc = 100. * correct / total
        print('acc is {}'.format(acc))
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer=nn.DataParallel(optimizer,device_ids=device_ids)
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "n_gen": args.n_gen,
    }

    writer = SummaryWriter()
    updater = BANUpdater_ensemble(**kwargs)
    criterion = nn.CrossEntropyLoss()

    i = 0
    best_loss = 1e+9
    best_loss_list = []

    print("train...")

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print('Successfully make dir {}'.format(args.outdir))

    last_model_weight_lst=[]

    for gen in range(args.resume_gen, args.n_gen):
        if gen == 0 or gen == 1 :
            last_model_weight=train(args,device,writer,train_loader,test_loader,gen,criterion,updater,i,logger)
        if gen > 1:
            # train BAN-2 for 10 epochs with initial w
            last_model_weight = train(args, device, writer, train_loader, test_loader, gen, criterion, updater, i,logger)
            # fix theta_2 and train w


        last_model_weight_lst.append(last_model_weight)
        print("best loss: ", best_loss)
        print("Born Again...")
        # load weights from saved snapshot
        updater.register_last_model(last_model_weight_lst,device_ids)
        updater.gen += 1
        best_loss_list.append(best_loss)
        best_loss = 1e+9
        # initialize self (mode and optimizer)
        model = config.get_model().to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        updater.model = model
        updater.optimizer = optimizer

    for gen in range(args.n_gen):
        print("Gen: ", gen,
              ", best loss: ", best_loss_list[gen])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args,device,writer,train_loader,test_loader,gen,criterion,updater,i,logger):
    for epoch in range(args.n_epoch):
        train_loss = 0
        ce_loss = 0
        kld_loss = 0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if gen == 0:
                t_loss = updater.update(inputs, targets, criterion).item()
                temp_ce_loss = t_loss
                temp_kld_loss = 0
            else:
                temp_ce_loss, temp_kld_loss = updater.update(inputs, targets, criterion)
                t_loss = (temp_ce_loss + temp_kld_loss).item()
            train_loss += t_loss
            ce_loss += temp_ce_loss
            kld_loss += temp_kld_loss
            i += 1
            if i % args.print_interval == 0:
                writer.add_scalar("train_loss", train_loss / args.print_interval, i)
                writer.add_scalar("ce_loss", ce_loss / args.print_interval, i)
                writer.add_scalar("kld_loss", kld_loss / args.print_interval, i)
                val_loss = 0
                with torch.no_grad():
                    for idx, (inputs, targets) in enumerate(test_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = updater.model(inputs)
                        loss = criterion(outputs, targets).item()
                        val_loss += loss

                val_loss /= len(test_loader)
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save weights
                    last_model_weight = os.path.join(args.outdir,
                                                     "model" + str(gen) + ".pth.tar")
                    torch.save(updater.model.module.state_dict(),
                               last_model_weight)  # add module aiming not to change the evalutae process

                writer.add_scalar("val_loss", val_loss, i)

                logger.print_log(epoch, i, train_loss / args.print_interval, val_loss)
                train_loss = 0
                ce_loss = 0
                kld_loss = 0


    return last_model_weight



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
    main()
