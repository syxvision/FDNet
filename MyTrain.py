#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from FDNet import FDNet
from apex import amp
from data2 import get_loader
import numpy as np
import random
import torch.backends.cudnn as cudnn

def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)


    return (wbce+wiou).mean()

def bce_loss(pred,gt):
    eposion = 1e-10
    sigmoid_pred = torch.sigmoid(pred)
    count_pos = torch.sum(gt)*1.0+eposion
    count_neg = torch.sum(1.-gt)*1.0
    beta = count_neg/count_pos
    beta_back = count_pos / (count_pos + count_neg)

    bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
    loss = beta_back*bce1(pred, gt)

    return loss

def train(Dataset, Network):
    setup_seed()
    ## dataset
    cfg = Dataset.Config(datapath='../data/COD10K', savepath='./out/out_1', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=51, training=True)
    data = Dataset.Data(cfg)

    os.makedirs(cfg.savepath, exist_ok=True)
    train_root = '../data/Train/'
    fn_root = './result/train/FDNet/fn/'
    fp_root = './result/train/FDNet/fp/'

    loader = get_loader(image_root=train_root + 'Imgs/',
                        gt_root=train_root + 'GT/',
                        fn_root = fn_root,
                        fp_root = fp_root,
                        batchsize=cfg.batch,
                        trainsize=288,
                        num_workers=8)
    ## network
    net = Network(cfg, pretrain=True)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        for step, (image, image_2_0, mask,fn,fp) in enumerate(loader):
            image_2_0 = image_2_0.cuda().float()
            image, mask= image.cuda().float(), mask.cuda().float()
            fn, fp = fn.cuda().float(), fp.cuda().float()

            fn_p,fp_p,y= net([image, image_2_0])

            lossfn = bce_loss(fn_p, fn)
            lossfp = bce_loss(fp_p, fp)
            lossp = structure_loss(y,mask)

            loss = 10*lossfn + 10*lossfp + lossp

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss': loss.item()},global_step=global_step)

            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f| lossfn=%.6f| lossfp=%.6f | lossp=%.6f ' %
                      (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],loss.item(),lossfn.item(),lossfp.item(),lossp.item()))

        if epoch % 10 ==0:
            print('save model')
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))

if __name__=='__main__':
    print("gpu:", torch.cuda.current_device())
    train(dataset, FDNet)

