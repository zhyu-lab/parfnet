import os

import cv2

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DATASET:
    def __init__(self,
                 im_path,
                 mask_path,
                 train_transform,
                 val_transform,
                 experiment='MoNuSeg/',
                 num_classes=1,
                 shape=(224, 224),
                 shuffle=True,
                 debug=False, 
                 batch_size=4):
        self.train_transform = train_transform
        self.val_transform = val_transform

        if experiment == 'dsb2018/':
            train_ids, test_ids = train_test_split(np.arange(os.listdir(im_path).__len__()),
                                                   test_size=0.25,
                                                   random_state=42,
                                                   shuffle=shuffle)
            if debug:
                train_ids, test_ids = train_test_split(np.arange(int(batch_size * 2)),
                                                       test_size=0.5,
                                                       random_state=42,
                                                       shuffle=shuffle)

            self.train_dataset = DSB(image_dir=im_path,
                                     mask_dir=mask_path,
                                     num_classes=num_classes,
                                     indexes=train_ids,
                                     shape=shape,
                                     transform=self.train_transform
                                     )
            self.test_dataset = DSB(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
                                    )

        elif experiment == 'GlaS/':
            im_dir = os.listdir(im_path)
            train_ids = [i for i, d in enumerate(im_dir) if d[:5] == 'train']
            test_ids = [i for i, d in enumerate(im_dir) if d[:4] == 'test']
            
            if debug:
                train_ids, test_ids = train_test_split(np.arange(2),
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=shuffle)

            self.train_dataset = GlaS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=train_ids,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = GlaS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
            )

        elif experiment == 'MoNuSeg/':
            train_im_dir = im_path + 'train_images'
            test_im_dir = im_path + 'test_images'
            train_mask_dir = mask_path + 'train_masks'
            test_mask_dir = mask_path + 'test_masks'

                            
            self.train_dataset = MoNuSeg(image_dir=train_im_dir,
                                    mask_dir=train_mask_dir,
                                    num_classes=num_classes,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = MoNuSeg(image_dir=test_im_dir,
                                    mask_dir=test_mask_dir,
                                    num_classes=num_classes,
                                    shape=shape,
                                    transform=self.val_transform
            )

        else:
            raise Exception('Dataset not found.')

        print('Data load completed.')

class GlaS(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(224, 224),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]        
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item][:-4] + '_anno.bmp')
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask[mask > self.num_classes] = self.num_classes
            mask[mask < 0] = 0
        return image, mask.long()

    def __len__(self):
        return len(self.im_list)
class MoNuSeg(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 shape=(224, 224),
                 transform=None,
                 ):
        self.im_dir = image_dir
        print(self.im_dir)
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        self.ToTensor = transforms.ToTensor()
        self.im_list = os.listdir(self.im_dir)

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item].replace('tif', 'png'))
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask[mask > self.num_classes] = self.num_classes
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)
class DSB(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(224, 224),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (224, 224))

        image = Image.fromarray(image.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.astype('uint8'), 'L')



        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            if self.num_classes == 1:
                mask[mask >= self.num_classes] = self.num_classes
            else:
                mask[mask > self.num_classes-1] = self.num_classes-1
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)
