import os
import sys
import numpy as np
import torch

from torch.utils import data

from skimage import io
from skimage import color
from skimage import transform
from skimage import util

# Class that reads a sequence of image paths from a directory and creates a data.Dataset with them.
class ListDataset(data.Dataset):

    def __init__(self, dataset, mode, crop_size, normalization='minmax', hidden_classes=None, overlap=False, use_dsm=False):

        # Initializing variables.
        self.root = './' + dataset + '/'
        self.dataset = dataset
        self.mode = mode
        self.crop_size = crop_size
        self.normalization = normalization
        self.hidden_classes = hidden_classes
        self.overlap = overlap
        self.use_dsm = use_dsm
        
        self.num_classes = 5 # For Vaihingen and Potsdam.
            
        if self.hidden_classes is not None:
            self.n_classes = self.num_classes - len(hidden_classes)
        else:
            self.n_classes = self.num_classes

        # Creating list of paths.
        self.imgs = self.make_dataset()

        # Check for consistency in list.
        if len(self.imgs) == 0:

            raise (RuntimeError('Found 0 images, please check the data set'))

    def make_dataset(self):

        # Making sure the mode is correct.
        assert self.mode in ['Train', 'Test', 'Validate']

        # Setting string for the mode.
        img_dir = os.path.join(self.root, self.mode, 'JPEGImages')
        msk_dir = os.path.join(self.root, self.mode, 'Masks')
        if self.use_dsm:
            dsm_dir = os.path.join(self.root, self.mode, 'NDSM')

        if self.mode == 'Validate':
            img_dir = os.path.join(self.root, 'Train', 'JPEGImages')
            msk_dir = os.path.join(self.root, 'Train', 'Masks')
            if self.use_dsm:
                dsm_dir = os.path.join(self.root, 'Train', 'NDSM')

        data_list = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

        # Creating list containing image and ground truth paths.
        items = []
        if self.dataset == 'Vaihingen':
            for it in data_list:
                item = (
                    os.path.join(img_dir, it),
                    os.path.join(msk_dir, it),
                    os.path.join(dsm_dir, it.replace('top_mosaic_09cm_area', 'dsm_09cm_matching_area').replace('.tif', '_normalized.jpg'))
                )
                items.append(item)
        elif self.dataset == 'Potsdam':
            for it in data_list:
                item = (
                    os.path.join(img_dir, it),
                    os.path.join(msk_dir, it.replace('_IRRG.tif', '_label_noBoundary.tif')),
                    os.path.join(dsm_dir, it.replace('top_potsdam_', 'dsm_potsdam_').replace('_IRRG.tif', '_normalized_lastools.jpg'))
                )
                items.append(item)
        
        # Returning list.
        return items
    
    def random_crops(self, img, msk, msk_true, n_crops):
        
        img_crop_list = []
        msk_crop_list = []
        msk_true_crop_list = []
        
        rand_fliplr = np.random.random() > 0.50
        rand_flipud = np.random.random() > 0.50
        rand_rotate = np.random.random()
        
        for i in range(n_crops):
            
            rand_y = np.random.randint(msk.shape[0] - self.crop_size[0])
            rand_x = np.random.randint(msk.shape[1] - self.crop_size[1])

            img_patch = img[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_patch = msk[rand_y:(rand_y + self.crop_size[0]),
                            rand_x:(rand_x + self.crop_size[1])]
            msk_true_patch = msk_true[rand_y:(rand_y + self.crop_size[0]),
                                      rand_x:(rand_x + self.crop_size[1])]
            
            if rand_fliplr:
                img_patch = np.fliplr(img_patch)
                msk_patch = np.fliplr(msk_patch)
                msk_true_patch = np.fliplr(msk_true_patch)
            if rand_flipud:
                img_patch = np.flipud(img_patch)
                msk_patch = np.flipud(msk_patch)
                msk_true_patch = np.flipud(msk_true_patch)
            
            if rand_rotate < 0.25:
                img_patch = transform.rotate(img_patch, 270, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 270, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 270, order=0, preserve_range=True)
            elif rand_rotate < 0.50:
                img_patch = transform.rotate(img_patch, 180, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 180, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 180, order=0, preserve_range=True)
            elif rand_rotate < 0.75:
                img_patch = transform.rotate(img_patch, 90, order=1, preserve_range=True)
                msk_patch = transform.rotate(msk_patch, 90, order=0, preserve_range=True)
                msk_true_patch = transform.rotate(msk_true_patch, 90, order=0, preserve_range=True)
                
            img_patch = img_patch.astype(np.float32)
            msk_patch = msk_patch.astype(np.int64)
            msk_true_patch = msk_true_patch.astype(np.int64)
            
            img_crop_list.append(img_patch)
            msk_crop_list.append(msk_patch)
            msk_true_crop_list.append(msk_true_patch)
        
        img = np.asarray(img_crop_list)
        msk = np.asarray(msk_crop_list)
        msk_true = np.asarray(msk_true_crop_list)
        
        return img, msk, msk_true
        
    def test_crops(self, img, msk, msk_true):
        
        n_channels = 3
        if self.use_dsm:
            n_channels = 4
        if self.overlap:
            w_img = util.view_as_windows(img,
                                         (self.crop_size[0], self.crop_size[1], n_channels),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2, n_channels)).squeeze()
            w_msk = util.view_as_windows(msk,
                                         (self.crop_size[0], self.crop_size[1]),
                                         (self.crop_size[0] // 2, self.crop_size[1] // 2))
            w_msk_true = util.view_as_windows(msk_true,
                                              (self.crop_size[0], self.crop_size[1]),
                                              (self.crop_size[0] // 2, self.crop_size[1] // 2))
        else:
            w_img = util.view_as_blocks(img, (self.crop_size[0], self.crop_size[1], n_channels)).squeeze()
            w_msk = util.view_as_blocks(msk, (self.crop_size[0], self.crop_size[1]))
            w_msk_true = util.view_as_blocks(msk_true, (self.crop_size[0], self.crop_size[1]))
        
        return w_img, w_msk, w_msk_true
        
    def shift_labels(self, msk):
        
        msk_true = np.copy(msk)
        
        cont = 0
        for h_c in self.hidden_classes:
            
            msk[msk == h_c - cont] = 100
            for c in range(h_c - cont + 1, self.num_classes):
                msk[msk == c] = c - 1
                msk_true[msk_true == c] = c - 1
            cont = cont + 1
        
        msk_true[msk == 100] = self.num_classes - len(self.hidden_classes)
        msk[msk == 100] = self.num_classes
        
        return msk, msk_true
    
    def mask_to_class(self, msk):
    
        msk = msk.astype(np.int64)
        new = np.zeros((msk.shape[0], msk.shape[1]), dtype=np.int64)
        
        msk = msk // 255
        msk = msk * (1, 7, 49)
        msk = msk.sum(axis=2)

        new[msk == 1 + 7 + 49] = 0 # Street.
        new[msk ==         49] = 1 # Building.
        new[msk ==     7 + 49] = 2 # Grass.
        new[msk ==     7     ] = 3 # Tree.
        new[msk == 1 + 7     ] = 4 # Car.
        new[msk == 1         ] = 5 # Surfaces.
        new[msk == 0         ] = 6 # Boundaries.

        return new
        
    def __getitem__(self, index):
        
        # Reading items from list.
        if self.use_dsm:
            img_path, msk_path, dsm_path = self.imgs[index]
        else:
            img_path, msk_path = self.imgs[index]
        
        # Reading images.
        img_raw = io.imread(img_path)
        msk_raw = io.imread(msk_path)
        if self.use_dsm:
            dsm_raw = io.imread(dsm_path)
            
        if len(img_raw.shape) == 2:
            img_raw = color.gray2rgb(img_raw)
        
        if self.use_dsm:
            img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                           img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                           img_raw.shape[2] + 1),
                          fill_value=0.0,
                          dtype=np.float32)
        else:
            img = np.full((img_raw.shape[0] + self.crop_size[0] - (img_raw.shape[0] % self.crop_size[0]),
                           img_raw.shape[1] + self.crop_size[1] - (img_raw.shape[1] % self.crop_size[1]),
                           img_raw.shape[2]),
                          fill_value=0.0,
                          dtype=np.float32)
        
        msk = np.full((msk_raw.shape[0] + self.crop_size[0] - (msk_raw.shape[0] % self.crop_size[0]),
                       msk_raw.shape[1] + self.crop_size[1] - (msk_raw.shape[1] % self.crop_size[1]),
                       msk_raw.shape[2]),
                      fill_value=0,
                      dtype=np.int64)
        
        img[:img_raw.shape[0], :img_raw.shape[1], :img_raw.shape[2]] = img_raw
        if self.use_dsm:
            img[:dsm_raw.shape[0], :dsm_raw.shape[1], -1] = dsm_raw
        msk[:msk_raw.shape[0], :msk_raw.shape[1]] = msk_raw
        
        msk = self.mask_to_class(msk)
        
        msk, msk_true = self.shift_labels(msk)
        
        # Normalization.
        img = (img / 255) - 0.5
        
        if self.mode == 'Train':
            
            img, msk, msk_true = self.random_crops(img, msk, msk_true, 3)
            
            img = np.transpose(img, (0, 3, 1, 2))
        
        elif self.mode == 'Validate':
            
            img, msk, msk_true = self.test_crops(img, msk, msk_true)
            
            img = np.transpose(img, (0, 1, 4, 2, 3))
            msk = np.transpose(msk, (0, 1, 2, 3))
            msk_true = np.transpose(msk_true, (0, 1, 2, 3))
        
        elif self.mode == 'Test':
            
            img, msk, msk_true = self.test_crops(img, msk, msk_true)
            
            img = np.transpose(img, (0, 1, 4, 2, 3))
            msk = np.transpose(msk, (0, 1, 2, 3))
            msk_true = np.transpose(msk_true, (0, 1, 2, 3))
        
        msk[msk == self.num_classes + 1] = self.num_classes
        msk_true[msk_true == self.num_classes + 1] = self.num_classes

        # Splitting path.
        spl = img_path.split('/')

        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)
        msk_true = torch.from_numpy(msk_true)

        # Returning to iterator.
        return img, msk, msk_true, spl[-1]

    def __len__(self):

        return len(self.imgs)
