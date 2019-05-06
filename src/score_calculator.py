import torch
from torch.utils.data import DataLoader
from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from dataset.pascal_voc import PascalVocDataset
from dataset.deepglobe import DeepGlobeDataset
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


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
            self.classes = np.arange(1, 21)
        elif dataset == 'cityscapes':
            self.valid_dataset = CityscapesDataset(split=split, net_type=net_type)
            self.classes = np.arange(1, 19)
        elif dataset == 'deepglobe':
            self.valid_dataset = DeepGlobeDataset(split=split, net_type=net_type)
            self.classes = np.arange(1, 7)
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
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def lovasz_softmax(logits, labels):
        probas = F.softmax(logits, dim=1)
        total_loss = 0
        batch_size = logits.shape[0]
        for prb, lbl in zip(probas, labels):
            total_loss += ScoreCalculator.lovasz_softmax_flat(prb, lbl, ignore_index=None, only_present=True)
        return total_loss / batch_size

    @staticmethod
    def lovasz_softmax_flat(prb, lbl, ignore_index, only_present):
        """
        Multi-class Lovasz-Softmax loss
          prb: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          lbl: [P] Tensor, ground truth labels (between 0 and C - 1)
          ignore_index: void class labels
          only_present: average only on classes present in ground truth
        """
        C = prb.shape[0]
        prb = prb.permute(1, 2, 0).contiguous().view(-1, C)  # H * W, C
        lbl = lbl.view(-1)  # H * W
        if ignore_index is not None:
            mask = lbl != ignore_index
            if mask.sum() == 0:
                return torch.mean(prb * 0)
            prb = prb[mask]
            lbl = lbl[mask]

        total_loss = 0
        cnt = 0
        for c in range(C):
            fg = (lbl == c).float()  # foreground for class c
            if only_present and fg.sum() == 0:
                continue
            errors = (fg - prb[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            total_loss += torch.dot(errors_sorted, ScoreCalculator.lovasz_grad(fg_sorted))
            cnt += 1
        # Change this line to fix a crash that occurs when cnt is 0
        # return total_loss / cnt
        return total_loss if cnt == 0 else total_loss / cnt

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
            if union != 0:
                ious.append(intersection / union)
        return ious if ious else [1]

    @staticmethod
    def compute_iou_batch(preds, labels, classes=None):
        iou = np.nanmean(
            [np.nanmean(ScoreCalculator.compute_ious(pred, label, classes, only_present=False)) for pred, label in zip(preds, labels)])
        return iou

    def compute_valid_loss_and_iou(self):
        valid_losses = []
        valid_ious = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.valid_loader) as _tqdm:
                for batched in _tqdm:
                    images, labels = batched
                    if self.fp16:
                        images = images.half()
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = self.model.tta(images, net_type=self.net_type)
                    if self.fp16:
                        loss = ScoreCalculator.lovasz_softmax(preds.float(), labels)
                    else:
                        loss = ScoreCalculator.lovasz_softmax(preds, labels)

                    preds_np = preds.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    # I changed a parameter in the compute_iou method to prevent it from yielding nans
                    iou = ScoreCalculator.compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, self.classes)

                    _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
                    valid_losses.append(loss.item())
                    valid_ious.append(iou)

        valid_loss = np.mean(valid_losses)
        valid_iou = np.mean(valid_ious)
        print(f'[Score Calculator] valid seg loss: {valid_loss}')
        print(f'[Score Calculator] valid iou: {valid_iou}')


if __name__ == '__main__':
    print('[Score Calculator] Starting computation')
    score_calculator = ScoreCalculator()
    score_calculator.compute_valid_loss_and_iou()

