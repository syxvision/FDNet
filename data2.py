import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2
import albumentations as A

from transforms.resize import ms_resize, ss_resize
from transforms.rotate import UniRotate
from transforms.image import read_color_array, read_gray_array

import torch
# several data augumentation strategies
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,fn_root,fp_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.fns = [fn_root + f for f in os.listdir(fn_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.fps = [fp_root + f for f in os.listdir(fp_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fns = sorted(self.fns)
        self.fps = sorted(self.fps)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )

    def __getitem__(self, index):

        image = read_color_array(self.images[index])
        mask = read_gray_array(self.gts[index], to_normalize=True, thr=0.5)
        fn = read_gray_array(self.fns[index], to_normalize=True, thr=0.5)
        fp = read_gray_array(self.fps[index], to_normalize=True, thr=0.5)
        transformed = self.joint_trans(image=image, mask=mask, fn=fn, fp=fp)
        image = transformed["image"]
        mask = transformed["mask"]
        fn = transformed['fn']
        fp = transformed['fp']

        base_h = 288
        base_w = 288
        images = ms_resize(image, scales=(1.0, 2.0), base_h=base_h, base_w=base_w)
        image_1_0 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_2_0 = torch.from_numpy(images[1]).permute(2, 0, 1)

        mask = ss_resize(mask, scale=2.0, base_h=base_h, base_w=base_w)
        mask_2_0 = torch.from_numpy(mask).unsqueeze(0)

        fn = ss_resize(fn, scale=2.0, base_h=base_h, base_w=base_w)
        fn = torch.from_numpy(fn).unsqueeze(0)

        fp = ss_resize(fp, scale=2.0, base_h=base_h, base_w=base_w)
        fp = torch.from_numpy(fp).unsqueeze(0)

        # return image, gt
        return image_1_0, image_2_0,mask_2_0,fn,fp

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images) and len(self.gts) ==len(self.fns) and len(self.fps) ==len(self.fns)
        images = []
        gts = []
        fps = []
        fns = []
        for img_path, gt_path ,fn_path,fp_path in zip(self.images, self.gts,self.fns,self.fps):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            fn = Image.open(fn_path)
            fp = Image.open(fp_path)
            if img.size == gt.size and gt.size == fn.size and fn.size == fp.size:
                images.append(img_path)
                gts.append(gt_path)
                fns.append(fn_path)
                fps.append(fp_path)
        self.images = images
        self.gts = gts
        self.fns = fns
        self.fps = fps


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root,fn_root,fp_root ,batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root,fn_root,fp_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def load_data(self):


        image = read_color_array(self.images[self.index])

        image = self.image_norm(image=image)["image"]

        base_h = 288
        base_w = 288
        images = ms_resize(image, scales=(1.0, 2.0), base_h=base_h, base_w=base_w)
        image_1_0 = torch.from_numpy(images[0]).permute(2, 0, 1).unsqueeze(0)
        image_2_0 = torch.from_numpy(images[1]).permute(2, 0, 1).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        #image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = self.binary_loader(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        return image_1_0,image_2_0, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
