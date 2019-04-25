import numpy as np
import matplotlib
import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from dataset.pascal_voc import PascalVocDataset
from dataset.deepglobe import DeepGlobeDataset
from utils.preprocess import minmax_normalize


class Tester:
    def __init__(self, model_path='../model/deepglobe_deeplabv3/model_tmp.pth', dataset='deepglobe',
                 output_channels=19, split='valid', net_type='deeplab', batch_size=1, shuffle=True):

        print('[Tester] [Init] Initializing tester...')

        matplotlib.use('Agg')
        self.model_path = model_path

        # Load model
        print('[Tester] [Init] Loading model ' + model_path + ' with '+str(output_channels)+' output channels...')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SPPNet(output_channels=output_channels).to(self.device)
        param = torch.load(model_path)
        self.model.load_state_dict(param)
        del param

        # Create data loader depending on dataset, split and net type
        if dataset == 'pascal':
            self.valid_dataset = PascalVocDataset(split=split, net_type=net_type)
        elif dataset == 'cityscapes':
            self.valid_dataset = CityscapesDataset(split=split, net_type=net_type)
        elif dataset == 'deepglobe':
            self.valid_dataset = DeepGlobeDataset(split=split, net_type=net_type)
        else:
            raise NotImplementedError

        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle)

        print('[Tester] [Init] ...done!')
        print('[Tester] [Init] Tester created.')

    def make_demo_image(self):
        images_list = []
        labels_list = []
        preds_list = []

        print('[Tester] [Demo] Gathering images and inferring...')
        self.model.eval()
        with torch.no_grad():
            for batched in self.valid_loader:
                images, labels = batched
                images_np = images.numpy().transpose(0, 2, 3, 1)
                labels_np = labels.numpy()

                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model.tta(images, net_type='deeplab')
                preds = preds.argmax(dim=1)
                preds_np = preds.detach().cpu().numpy()

                images_list.append(images_np)
                labels_list.append(labels_np)
                preds_list.append(preds_np)

                if len(images_list) == 4:
                    break

        print('[Tester] [Demo] Processing results...')

        images = np.concatenate(images_list)
        labels = np.concatenate(labels_list)
        preds = np.concatenate(preds_list)

        # Ignore index
        ignore_pixel = labels == 255
        preds[ignore_pixel] = 0
        labels[ignore_pixel] = 0

        # Plot
        fig, axes = plt.subplots(4, 3, figsize=(12, 10))
        plt.tight_layout()

        axes[0, 0].set_title('input image')
        axes[0, 1].set_title('prediction')
        axes[0, 2].set_title('ground truth')

        for ax, img, lbl, pred in zip(axes, images, labels, preds):
            ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
            ax[1].imshow(pred)
            ax[2].imshow(lbl)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].set_xticks([])
            ax[2].set_yticks([])

        plt.savefig('eval.png')
        plt.close()

        print('[Tester] [Demo] Done.')

    def infer_image_by_path(self, image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/test1.jpg'):
        # Notify layers that we are in eval mode (for batchnorm, dropout)
        self.model.eval()
        # Deactivate autograd engine to reduce memory usage (no need backprop when inferring)
        with torch.no_grad():
            print('[Tester] [Single test] Preparing image ' + image_path + '...')
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Transpose image to fit torch tensor format
            img = img.transpose(2, 0, 1)
            # Conversion to torch tensor
            img_tensor = torch.tensor([img])
            # Conversion to float to fit torch model
            img_tensor = img_tensor.float()

            print('[Tester] [Single test] Inferring...')

            # Send to CPU or GPU depending on the hardware found
            img_tensor = img_tensor.to(self.device)

            # Generate predictions
            preds = self.model.tta(img_tensor, net_type='deeplab')
            # Not sure what this does in initial code, probably helps extracting a single class
            preds = preds.argmax(dim=1)

            # Convert back to nparray procssable as an image
            preds_np = preds.detach().cpu().numpy()
            preds_np = preds_np[0]

            print('[Tester] [Single test] Generating mask...')

            pred_pil = Image.fromarray(preds_np.astype(np.uint8))
            pred_pil.show()

            print('[Tester] [Single test] Done.')


if __name__ == '__main__':
    # model_path = '../model/pascal_deeplabv3p_with_pretrained/model.pth'
    tester = Tester(model_path='../model/deepglobe_deeplabv3/model_tmp.pth')
    tester.infer_image_by_path()
    tester.infer_image_by_path(image_path='../data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_000549.jpg')
    tester.make_demo_image()
