import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize
from utils.custum_aug import PadIfNeededRightBottom

from utils.constants import color_correspondences, web_palette_values

import random

# This version of the dataset loaser will change the images picked at each initialization
# This is made to prevent overfitting that occurs if we do the tile balancing by picking always the
# Same ones


class DeepGlobeDatasetDynamic(Dataset):
    n_classes = 6

    def __init__(self, base_dir='../data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012', split='train',
                 affine_augmenter=None, image_augmenter=None, target_size=(512, 512),
                 net_type='unet', ignore_index=255, debug=False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = split
        self.target_size= target_size
        ######################################
        #          This will change :        #
        ######################################

        # Generate randomized valid split
        valid_ids = []
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'AgricultureLand' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'BarrenLand' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'Forest' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'RangeLand' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'UrbanLand' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]
        valid_ids_dir = self.base_dir / 'ClassifiedTiles' / 'Water' / 'val.txt'
        with open(valid_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            valid_ids = valid_ids + temp_ids[:69]

        # Generate randomized train split
        train_ids = []
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'AgricultureLand' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'BarrenLand' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'Forest' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'RangeLand' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'UrbanLand' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]
        train_ids_dir = self.base_dir / 'ClassifiedTiles' / 'Water' / 'train.txt'
        with open(train_ids_dir, 'r') as f:
            temp_ids = f.readlines()
            random.shuffle(temp_ids)
            train_ids = train_ids + temp_ids[:278]

        lbl_dir = 'SegmentationClass'
        if self.split == 'valid':
            img_ids = valid_ids
        elif self.split == 'train':
            img_ids = train_ids
        else:
            valid_set = set([valid_id.strip() for valid_id in valid_ids])
            lbl_dir = 'SegmentationClassAug' if 'aug' in split else 'SegmentationClass'
            all_set = set([p.name[:-4] for p in self.base_dir.joinpath(lbl_dir).iterdir()])
            img_ids = list(all_set - valid_set)
        self.img_paths = [(self.base_dir / 'JPEGImages' / f'{img_id.strip()}.jpg') for img_id in img_ids]
        self.lbl_paths = [(self.base_dir / lbl_dir / f'{img_id.strip()}.png') for img_id in img_ids]

        # Resize
        if isinstance(target_size, str):
            target_size = eval(target_size)
        if 'train' in self.split:
            if self.net_type == 'deeplab':
                target_size = (target_size[0] + 1, target_size[1] + 1)
            #self.resizer = albu.Compose([albu.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),
            #                             PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1],
            #                                                    value=0, ignore_index=self.ignore_index, p=1.0),
            #                             albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
            self.resizer = albu.Compose([albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
        else:
            # self.resizer = None
            self.resizer = albu.Compose([PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1],
                                                                value=0, ignore_index=self.ignore_index, p=1.0),
                                         albu.Crop(x_min=0, x_max=target_size[1],
                                                   y_min=0, y_max=target_size[0])])

        # Augment
        if 'train' in self.split:
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        # For debug
        # print('[LOADER] Got image ' + str(img_path))
        img = np.array(Image.open(img_path))
        if self.split == 'test':
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img)
                img = resized['image']
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            return img
        else:
            lbl_path = self.lbl_paths[index]

            inter_img = Image.open(lbl_path)
            # Convert the RGB images to the P mode in PIL
            # inter_img = inter_img.convert('P', palette=Image.ADAPTIVE, colors=256) C'EST DE LA MERDE
            inter_img = inter_img.convert('P', palette=Image.WEB)

            self.palette_to_indexes(inter_img)

            lbl = np.array(inter_img)
    
            # unique, counts = np.unique(lbl, return_counts=True)
            # n_max= unique[(counts == counts.max())][0]
            
            
            # lbl = (lbl == n_max)*(n_max+1) + (lbl == 255) * 255
            #lbl = lbl.astype(np.uint8)

            
            # ImageAugment (RandomBrightness, AddNoise...)
            if self.image_augmenter:
                augmented = self.image_augmenter(image=img)
                img = augmented['image']
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img, mask=lbl)
                img, lbl = resized['image'], resized['mask']
            # AffineAugment (Horizontal Flip, Rotate...)
            if self.affine_augmenter:
                augmented = self.affine_augmenter(image=img, mask=lbl)
                img, lbl = augmented['image'], augmented['mask']

            unique, counts = np.unique(lbl, return_counts=True)
            n_max= unique[(counts == counts.max())][0]
            lbl = (lbl == n_max)*(n_max+1) + (lbl == 255) * 255

            if self.debug:
                print(lbl_path)
                print(lbl.shape)
                print(np.unique(lbl))
            else:
                img = img.transpose(2, 0, 1)
                img = torch.FloatTensor(img)
                lbl = torch.LongTensor(lbl)
            return img, lbl

    @staticmethod
    def index_to_palette(pil_image):
        pixels = pil_image.load()  # create the pixel map

        for i_index in range(pil_image.size[0]):
            for j_index in range(pil_image.size[1]):
                if pixels[i_index, j_index] == color_correspondences['black']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['black']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['green']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['green']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['yellow']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['yellow']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['blue']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['blue']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['magenta']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['magenta']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['cyan']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['cyan']['web-palette'])
                elif pixels[i_index, j_index] == color_correspondences['white']['index']:
                    pil_image.putpixel((i_index, j_index), color_correspondences['white']['web-palette'])
                else:
                    pil_image.putpixel((i_index, j_index), color_correspondences['black']['web-palette'])

        pil_image.putpalette(web_palette_values)

        return pil_image

    @staticmethod
    def palette_to_indexes(pil_image):
        # Then I need to convert these values to a scale of 1..7
        pixels = pil_image.load()  # create the pixel map

        for i_index in range(pil_image.size[0]):
            for j_index in range(pil_image.size[1]):
                if pixels[i_index, j_index] == color_correspondences['black']['web-palette']:  # BLACK
                    pixels[i_index, j_index] = color_correspondences['black']['index']  # Unknown
                elif pixels[i_index, j_index] == color_correspondences['green']['web-palette']:  # GREEN
                    pixels[i_index, j_index] = color_correspondences['green']['index']  # Forest land
                elif pixels[i_index, j_index] == color_correspondences['yellow']['web-palette']:  # YELLOW
                    pixels[i_index, j_index] = color_correspondences['yellow']['index']  # Agriculture land
                elif pixels[i_index, j_index] == color_correspondences['blue']['web-palette']:  # BLUE
                    pixels[i_index, j_index] = color_correspondences['blue']['index']  # Water
                elif pixels[i_index, j_index] == color_correspondences['magenta']['web-palette']:  # MAGENTA
                    pixels[i_index, j_index] = color_correspondences['magenta']['index']  # Rangeland
                elif pixels[i_index, j_index] == color_correspondences['cyan']['web-palette']:  # CYAN
                    pixels[i_index, j_index] = color_correspondences['cyan']['index']  # Urban land
                elif pixels[i_index, j_index] == color_correspondences['white']['web-palette']:  # WHITE
                    pixels[i_index, j_index] = color_correspondences['white']['index']  # Barren land
                else:
                    exit(-1)

        return pil_image