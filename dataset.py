import torchvision
import torchvision.transforms as transforms
import PIL
import torch as t
import numpy as np
import lmdb
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import io
import string
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

def num_samples(dataset, train):
    if dataset == 'celeba256':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    # elif dataset == 'celeba64':
    #     return 50000 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

class LMDBDataset(t.utils.data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return num_samples(self.name, self.train)

def get_dataset(args):
    img_size = args.img_size

    if args.dataset == 'cifar10':
        data_dir = '/data4/jcui7/images/data/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/'
        print(data_dir)
        if args.normalize_data:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        ds_train = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=True, transform=transform)
        ds_val = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=False, transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape

    elif args.dataset == 'svhn':
        data_dir = '/data4/jcui7/images/data/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/'
        if args.normalize_data:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        ds_train = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='extra', transform=transform)
        ds_val = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='test', transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape

    elif args.dataset == 'svhn_fid':
        data_dir = '/data4/jcui7/images/data/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/'
        if args.normalize_data:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        ds_train = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='train', transform=transform)
        ds_val = torchvision.datasets.SVHN(data_dir + 'svhn', download=True, split='test', transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape

    if args.dataset == 'celeba64':
        num_classes = 40
        data_dir = '/data4/jcui7/HugeData/celeba64_org/celeba64_lmdb/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/celeba64_org/celeba64_lmdb'

        class CropCelebA64(object):
            """ This class applies cropping for CelebA64. This is a simplified implementation of:
            https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
            """

            def __call__(self, pic):
                new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
                return new_pic

            def __repr__(self):
                return self.__class__.__name__ + '()'

        train_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valid_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LMDBDataset(root=data_dir, name='celeba64', train=True, transform=train_transform,
                                 is_encoded=True)
        valid_data = LMDBDataset(root=data_dir, name='celeba64', train=False, transform=valid_transform,
                                 is_encoded=True)
        input_shape = [3, img_size, img_size]
        return train_data, valid_data, input_shape


