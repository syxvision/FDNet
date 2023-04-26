
import numpy as np
import os, argparse
import imageio.v2 as imageio
import cv2
import torch
from data2 import test_dataset
from FDNet import FDNet as Network
import dataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
testsize = 288 #testsize
method = 'FDNet'
for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
    data_path = '../data/Test/{}/'.format(_data_name)
    save_path = './result/result_1/{}/{}/'.format(method,_data_name)
    cfg = dataset.Config(datapath=data_path, snapshot='./out/out_1/model-51', mode='test')

    net = Network(cfg)
    net.train(False)
    net.cuda()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root,testsize)

    for i in range(test_loader.size):
        image_1_0, image_2_0, gt,name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        img_for_post = np.asarray(img_for_post, np.float32)
        img_for_post /= (img_for_post.max() + 1e-8)

        image_1_0 = image_1_0.cuda()
        image_2_0 = image_2_0.cuda()

        _,_,predict= net([image_1_0,image_2_0], gt.shape)
        res = predict

        res = torch.nn.functional.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite(save_path+name,(res*255).astype(np.uint8))
