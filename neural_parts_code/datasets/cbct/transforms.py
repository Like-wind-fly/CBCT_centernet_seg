"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize

#----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, scale):
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        self.scale = scale

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=(1,self.scale,self.scale),mode='trilinear', align_corners=False, recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)
        return img[0], mask[0]

class RandomResize:
    def __init__(self,s_rank, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0],self.w_rank[1])
        random_h = random.randint(self.h_rank[0],self.h_rank[1])
        random_s = random.randint(self.s_rank[0],self.s_rank[1])
        self.shape = [random_s,random_h,random_w]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape,mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].long()

class RandomCrop:
    def __init__(self, slicelists):
        self.slicelists =  slicelists

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):
        slicerange = []
        for i in range(1,4):
            ss, es = self._get_range(mask.size(i), self.slicelists[i-1])
            slicerange.append([ss, es])
        
        # print(self.shape, img.shape, mask.shape)
        tmp_img = img[:,slicerange[0][0]:slicerange[0][1],slicerange[1][0]:slicerange[1][1],slicerange[2][0]:slicerange[2][1]].clone()
        tmp_mask = mask[:,slicerange[0][0]:slicerange[0][1],slicerange[1][0]:slicerange[1][1],slicerange[2][0]:slicerange[2][1]].clone()
        return tmp_img, tmp_mask

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandFlip_ALL:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        flip_axis = []
        for i in range(1,4):
            prob = random.uniform(0, 1)
            if prob <= self.prob:
                flip_axis.append(i)

        if len(flip_axis)>0:
            img = img.flip(flip_axis)
            mask = mask.flip(flip_axis)

        return img, mask

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)

class RandomRotate_3d():
    def __init__(self):
        pass
    def __call__(self, img, mask):
        begin = [0]
        order = [1,2,3]
        random.shuffle(order)
        order = begin+order
        #
        img = img.permute(order)
        mask = mask.permute(order)

        angle = random.randint(-180,180)
        img = transforms.functional.rotate(img,angle,InterpolationMode.BILINEAR)
        mask = transforms.functional.rotate(mask, angle, InterpolationMode.NEAREST)

        return img,mask

class RandomPermute_3d():
    def __init__(self):
        pass
    def __call__(self, img, mask):
        begin = [0]
        order = [1,2,3]
        random.shuffle(order)
        order = begin+order

        img = img.permute(order)
        mask = mask.permute(order)
        return img,mask

class RandomRotate_2d():
    def __init__(self,maxrange =10):
        self.maxrange = maxrange
        pass
    def __call__(self, img, mask):
        angle = random.randint(-1*self.maxrange,self.maxrange)
        img = transforms.functional.rotate(img,angle,InterpolationMode.BILINEAR)
        mask = transforms.functional.rotate(mask, angle, InterpolationMode.BILINEAR)
        return img,mask

class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  # base默认取16，因为4次下采样后为1
        self.max_size = max_size 
        if self.max_size%self.base:
            self.max_size = self.max_size - self.max_size%self.base # max_size为限制最大采样slices数，防止显存溢出，同时也应为16的倍数
    def __call__(self, img , label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1)//2 - slice_num//2
        right =  img.size(1)//2 + slice_num//2

        crop_img = img[:,left:right]
        crop_label = label[:,left:right]
        return crop_img, crop_label

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

if __name__ == "__main__":
    t = RandomCrop([48,128,128])
    img = torch.zeros([1,160,256,256])
    mask= torch.zeros([1, 160, 256, 256])
    ct,label = t(img,mask)
    print(ct.shape)
    # print(ct.label)
