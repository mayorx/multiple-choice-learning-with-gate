import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import torchvision
import torchvision.transforms as transforms

from models import *


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--model-num', default='2', type=int, metavar='MN', help='the number of models')
parser.add_argument('--gate-type', default='1', type=int, metavar='GT', help='the type of gate')
parser.add_argument('--name', default='anonymous', type=str, metavar='NAME', help='the name of this run')

best_prec = 0
now_learning_rate = 0

def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !

        # model = resnet20_cifar()
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()
        # model = resnet164_cifar(num_classes=100)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)
        
        #model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/dcl-cifar-{}-mn-{}-gt-{}-{}'.format(args.cifar_type, args.model_num, args.gate_type, args.name)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        # if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
        #     model_type = 1
        # elif isinstance(model, Wide_ResNet_Cifar):
        #     model_type = 2
        # elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
        #     model_type = 3
        # else:
        #     print('model type unrecognized...')
        #     return
        gate = gate_factory(args.gate_type, args.model_num)
        models = []
        optimizers = []
        for i in range(0, args.model_num):
            model = resnet32_cifar(num_classes=args.cifar_type)
            model = nn.DataParallel(model).cuda()
            optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                  nesterov=True)
            models.append(model)
            optimizers.append(optimizer)

        gate = nn.DataParallel(gate).cuda()
        gate_optimizer = optim.SGD(gate.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                   nesterov=True)
        criterion = nn.CrossEntropyLoss(reduce=False).cuda()
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model_num = checkpoint['model_num']
            args.start_epoch = checkpoint['epoch']
            gate.load_state_dict(checkpoint['gate'])
            gate_optimizer.load_state_dict(checkpoint['gate_optimizer'])
            for i in range(model_num):
                models[i].load_state_dict(checkpoint['model-{}'.format(i)])
                optimizers[i].load_state_dict(checkpoint['optimizer-{}'.format(i)])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, models, gate, criterion)
        return

    print_important_args(args)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        for opti in optimizers:
            adjust_learning_rate(opti, epoch)
        adjust_learning_rate(gate_optimizer, epoch)

        print('Epoch: {0}\t LR = {lr:.4f}'.format(epoch, lr=now_learning_rate))
        # train for one epoch
        train(trainloader, criterion, models, optimizers, gate, gate_optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, models, gate, criterion)

        end_time = time.time()
        passed_time = end_time - start_time
        estimated_extra_time = passed_time * (args.epochs - epoch) / (epoch - args.start_epoch + 1)
        print('time flies very fast .. {passed_time:.2f} mins passed, about {extra:.2f} mins left.'.format(
            passed_time=passed_time / 60, extra=estimated_extra_time / 60))

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        save_checkpoint(epoch, args.model_num, models, optimizers, gate, gate_optimizer, fdir)

    print('finished. best_prec: {:.4f}'.format(best_prec))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(trainloader, criterion, models, optimizers, gate, gate_optimizer, epoch):
    model_num = len(models)

    for model in models:
        model.train()
    gate.train()

    losses = []
    top1 = []
    for idx in range(model_num):
        losses.append(AverageMeter())
        top1.append(AverageMeter())

    gate_pred_correct = 0
    for ix, (input, target) in enumerate(trainloader):
        input, target = input.cuda(), target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        outputs = [None] * model_num
        for idx in range(model_num):
            outputs[idx] = models[idx](input_var)

        pred = F.softmax(gate(input_var),dim=1)
        loss = 0

        losses_detail = pred.data.clone()

        for i in range(model_num):
            #f_loss = criterion(outputs[i], target_var) * pred[:, i].contiguous().view(-1, 1)
            f_loss = criterion(outputs[i], target_var)
            # print('train f loss: {}'.format(f_loss))
            # print('pred[:, {}] = {}'.format(i, pred[:, i]))
            # print('mul : {}'.format(f_loss * pred[:, i]))
            # print('mean: {}'.format((f_loss * pred[:, i]).mean()))

            losses_detail[:, i] = f_loss.data
            # print(f_loss)
            # print(pred[:, i])
            loss = loss + (f_loss * pred[:, i]).mean()
            prec = accuracy(outputs[i].data, target)[0]
            top1[i].update(prec[0], input.size(0))
            losses[i].update(f_loss.mean().data[0], input.size(0))

        _, min_loss_idx = losses_detail.topk(1, 1, False, True)
        _, max_pred_idx = pred.data.topk(1, 1, True, True)

        gate_pred_correct += (min_loss_idx == max_pred_idx).sum()


        for i in range(model_num):
            optimizers[i].zero_grad()
        gate_optimizer.zero_grad()
        loss.backward()
        gate_optimizer.step()
        for i in range(model_num):
            optimizers[i].step()

    for idx in range(model_num):
        # print(losses[idx].avg)
        print('model {0}\t Train: Loss {loss.avg:.4f} Prec {top1.avg:.3f}%'.format(idx, loss=losses[idx], top1=top1[idx]))

    print('gate predict correct Train {}/{} {:.2f}%\n\n'.format(gate_pred_correct, len(trainloader.dataset),100. * gate_pred_correct / len(trainloader.dataset)))

def validate(val_loader, models, gate, criterion):
    model_num = len(models)

    # switch to evaluate mode
    for model in models:
        model.eval()
    gate.eval()

    losses = []
    top1 = []
    for idx in range(model_num):
        losses.append(AverageMeter())
        top1.append(AverageMeter())

    total_top1 = AverageMeter()


    gate_pred_correct = 0
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        pred = gate(input_var)

        sample = i % 100 == 0
        # if sample:
        #     print('pred: {}'.format(pred.data))
        losses_detail = pred.data.clone()


        final_predicts = None
        for idx in range(model_num):
            output = models[idx](input_var)

            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec = accuracy(output.data, target)[0]
            losses[idx].update(loss.data[0], input.size(0))
            top1[idx].update(prec[0], input.size(0))


            losses_detail[:, idx] = loss.data

            tmp_predicts = F.softmax(output, dim=1) * pred[:, idx].contiguous().view(-1,1)
            if idx == 0:
                final_predicts = tmp_predicts
            else:
                final_predicts+= tmp_predicts

        # if sample:
        #     print('final_predicts: {}'.format(final_predicts))


        prec = accuracy(final_predicts.data, target)[0]
        total_top1.update(prec[0], input.size(0))

        _, min_loss_idx = losses_detail.topk(1, 1, False, True)
        _, max_pred_idx = pred.data.topk(1, 1, True, True)
        gate_pred_correct += (min_loss_idx == max_pred_idx).sum()


    for idx in range(model_num):
        print('model {0}\t Test: Loss {loss.avg:.4f} ,Prec {top1.avg:.3f}%'.format(idx, loss=losses[idx], top1=top1[idx]))

    print('mixture of experts result: Prec {top1.avg:.3f}%'.format(top1=total_top1))
    print('gate predict correct Test {}/{} {:.2f}%\n\n'.format(gate_pred_correct, len(val_loader.dataset),100. * gate_pred_correct / len(val_loader.dataset)))
    return total_top1.avg

# def save_checkpoint(state, is_best, fdir):
#     filepath = os.path.join(fdir, 'checkpoint.pth')
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))

def save_checkpoint(epoch, model_num, models, optimizers, gate, gate_optimizer, fdir):
    print('save checkpoint ... epoch {}, fdir {}'.format(epoch, fdir))
    filepath = os.path.join(fdir, 'checkpoint.pth')
    state = {
        'epoch' : epoch + 1,
        'model_num': model_num,
        'gate': gate.state_dict(),
        'gate_optimizer': gate_optimizer.state_dict(),
    }
    for i in range(model_num):
        state['model-{}'.format(i)] = models[i].state_dict()
        state['optimizer-{}'.format(i)] = optimizers[i].state_dict()
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global now_learning_rate
    if epoch < 60:
        lr = args.lr
    elif epoch < 120:
        lr = args.lr * 0.1
    elif epoch < 180:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001

    # """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # if model_type == 1:
    #     if epoch < 80:
    #         lr = args.lr
    #     elif epoch < 120:
    #         lr = args.lr * 0.1
    #     else:
    #         lr = args.lr * 0.01
    # elif model_type == 2:
    #     if epoch < 60:
    #         lr = args.lr
    #     elif epoch < 120:
    #         lr = args.lr * 0.2
    #     elif epoch < 160:
    #         lr = args.lr * 0.04
    #     else:
    #         lr = args.lr * 0.008
    # elif model_type == 3:
    #     if epoch < 150:
    #         lr = args.lr
    #     elif epoch < 225:
    #         lr = args.lr * 0.1
    #     else:
    #         lr = args.lr * 0.01
    now_learning_rate = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def gate_factory(gate_type, model_num):
    if gate_type == 1:
        raise('not support')
        # return GateNet(num_classes=model_num) #softmax as output
    elif gate_type == 2:
        raise('not support')
        # return gate_resnet(num_classes=model_num) #softmax as output
    elif gate_type == 3:
        return resnet32_cifar(num_classes=model_num) #regular output
    elif gate_type == 4:
        return GateNet(model_nums=model_num, sm=0) #regular output
    else:
        raise('gate type not found :{}'.format(gate_type))

def print_important_args(args):
    print('momentum {momentum} weight-decay {wd} batch-size {bs} model-num {mn} gate-type {gt} run-name {nm}'.format(momentum=args.momentum, wd=args.weight_decay, bs=args.batch_size, mn=args.model_num, gt=args.gate_type, nm=args.name))

if __name__=='__main__':
    main()

