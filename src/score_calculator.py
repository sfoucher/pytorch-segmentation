import torch
from torch.utils.data import DataLoader
from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from dataset.pascal_voc import PascalVocDataset
from dataset.deepglobe import DeepGlobeDataset
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from utils.constants import color_correspondences

from PIL import Image
import copy


class ScoreCalculator:
    def __init__(self, model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs/model.pth', dataset='deepglobe',
                 output_channels=19, split='valid', net_type='deeplab', batch_size=1, shuffle=True):
        print('[Score Calculator] Initializing calculator...')
        self.dataset = dataset
        self.model_path = model_path
        self.net_type = net_type

        self.fp16 = True

        # Load model
        print('[Score Calculator] Loading model ' + model_path + ' with ' + str(output_channels) + ' output channels...')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SPPNet(output_channels=output_channels).to(self.device)
        param = torch.load(model_path)
        self.model.load_state_dict(param)
        del param

        # Create data loader depending on dataset, split and net type
        if dataset == 'pascal':
            self.valid_dataset = PascalVocDataset(split=split, net_type=net_type)
            self.classes = np.arange(1, 22)
        elif dataset == 'cityscapes':
            self.valid_dataset = CityscapesDataset(split=split, net_type=net_type)
            self.classes = np.arange(1, 20)
        elif dataset == 'deepglobe':
            self.valid_dataset = DeepGlobeDataset(split=split, net_type=net_type)
            self.classes = np.arange(1, 8)
        else:
            raise NotImplementedError

        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle)

        # fp16
        if self.fp16:
            from utils.apex.apex.fp16_utils.fp16util import BN_convert_float
            self.model = BN_convert_float(self.model.half())
            print('[Score Calculator] fp16 applied')

        print('[Score Calculator] ...done!')
        print('[Score Calculator] Calculator created.')


    @staticmethod
    def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
        pred[label == ignore_index] = 0
        ious = []

        for c in classes:
            label_c = label == c
            if only_present and np.sum(label_c) == 0:
                ious.append(np.nan)
                continue
            pred_c = pred == c
            intersection = np.logical_and(pred_c, label_c).sum()
            union = np.logical_or(pred_c, label_c).sum()
            """
            if c == 0:
                print('0 :')
                print(intersection)
                print(union)
            if c == 10:
                print('10 :')
                print(intersection)
                print(union)
            """
            if float(union) != 0.0:
                ious.append(float(intersection) / float(union))
            else:
                ious.append(0.0)
        return ious if ious else [1]

    @staticmethod
    def compute_iou_batch(preds, labels, classes=None):
        iou = np.nanmean(
            [np.nanmean(ScoreCalculator.compute_ious(pred, label, classes, only_present=False)) for pred, label in zip(preds, labels)])
        return iou

    def compute_valid_loss_and_iou(self):
        debug = True
        valid_ious = []
        ious_by_class = []
        self.model.eval()
        i = 0
        with torch.no_grad():
            with tqdm(self.valid_loader) as _tqdm:
                for batched in _tqdm:
                    i += 1
                    images, labels = batched
                    if self.fp16:
                        images = images.half()
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = self.model.tta(images, net_type=self.net_type)
                    preds = preds.argmax(dim=1)
                    preds_np = preds.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    if debug:
                        preds_np_bis = copy.deepcopy(preds_np)
                        good_preds = preds_np_bis[0]
                        good_mask = Image.fromarray(good_preds.astype('uint8'), 'P')

                        # Transform mask to set good indexes and palette
                        good_mask = DeepGlobeDataset.index_to_palette(good_mask)
                        good_mask.save('../valid_masks/' + str(i) + '_prediction.png')

                    # I changed a parameter in the compute_iou method to prevent it from yielding nans

                    iou = ScoreCalculator.compute_iou_batch(preds_np, labels_np, self.classes)
                    ious_by_class.append(ScoreCalculator.compute_ious(preds_np, labels_np, self.classes))

                    _tqdm.set_postfix(OrderedDict(iou=f'{iou:.3f}'))
                    valid_ious.append(iou)

        # Compute mean ious by class
        iou_means_by_class = []
        if ious_by_class[0]:
            for i in range(len(ious_by_class[0])):
                val = 0.0
                num_val = 0.0
                for image in range(len(ious_by_class)):
                    if not np.isnan(ious_by_class[image][i]):
                        val += ious_by_class[image][i]
                        num_val += 1.0

                if num_val != 0.0:
                    iou_means_by_class.append((val / num_val))
                else:
                    print('No detection found for class ' + str(i))  # DEBUG
                    iou_means_by_class.append(0.0)

        i = 0
        for score in iou_means_by_class:
            i += 1
            if i == color_correspondences['black']['index']:
                print(color_correspondences['black']['description'] + ' : ' + str(score))
            elif i == color_correspondences['green']['index']:
                print(color_correspondences['green']['description'] + ' : ' + str(score))
            elif i == color_correspondences['yellow']['index']:
                print(color_correspondences['yellow']['description'] + ' : ' + str(score))
            elif i == color_correspondences['cyan']['index']:
                print(color_correspondences['cyan']['description'] + ' : ' + str(score))
            elif i == color_correspondences['magenta']['index']:
                print(color_correspondences['magenta']['description'] + ' : ' + str(score))
            elif i == color_correspondences['blue']['index']:
                print(color_correspondences['blue']['description'] + ' : ' + str(score))
            elif i == color_correspondences['white']['index']:
                print(color_correspondences['white']['description'] + ' : ' + str(score))

        # Don't count the unknown class
        print(f'[Score Calculator] Mean performance for all classes : {np.mean(iou_means_by_class[1:])}')


if __name__ == '__main__':
    print('[Score Calculator] Starting computation')
    score_calculator = ScoreCalculator(
        # model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs/model.pth')
        model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs_small-patches_dynamic/model.pth')
    score_calculator.compute_valid_loss_and_iou()

