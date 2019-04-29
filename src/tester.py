import numpy as np
import torch
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
        """
        Initializes the tester by loading the model with the good parameters.
        :param model_path: Path to model weights
        :param dataset: dataset used amongst {'deepglobe', 'pascal', 'cityscapes'}
        :param output_channels: num of output channels of model
        :param split: split to be used amongst {'train', 'valid'}
        :param net_type: model type to be used amongst {'deeplab', 'unet'}
        :param batch_size: batch size when loading images (always 1 here)
        :param shuffle: when loading images from dataset
        """

        print('[Tester] [Init] Initializing tester...')

        self.model_path = model_path

        # Load model
        print('[Tester] [Init] Loading model ' + model_path + ' with ' + str(output_channels) + ' output channels...')

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
        """
        Picks 4 images from dataset randomly and creates image with raw, inferred and label pictures.
        :return: null
        """
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
        plt.show()
        plt.close()

        print('[Tester] [Demo] Done.')

    def infer_image_by_path(self, image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/test1.jpg'):
        """
        Opens image from fs and passes it through the loaded network, then displays and saves the result.
        :param image_path: Path of input images
        :return: null
        """
        print('[Tester] [Single test] Opening image '+image_path+'...')
        custom_img = np.array(Image.open(image_path))
        custom_img = minmax_normalize(custom_img, norm_range=(-1, 1))
        custom_img = custom_img.transpose(2, 0, 1)
        custom_img = torch.FloatTensor([custom_img])

        print('[Tester] [Single test] Gathering images and inferring...')
        self.model.eval()
        with torch.no_grad():
            custom_img = custom_img.to(self.device)
            preds = self.model.tta(custom_img, net_type='deeplab')
            preds = preds.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()

        print('[Tester] [Single test] Processing results...')
        plt.imshow(preds_np[0])
        plt.savefig('single_test_output.png')
        plt.show()
        plt.close()

        print('[Tester] [Single test] Done.')


if __name__ == '__main__':
    # tester = Tester(model_path='../model/deepglobe_deeplabv3_second_pass/model_tmp.pth')
    tester = Tester(model_path='../model/pascal_deeplabv3p_with_pretrained/model.pth', dataset='pascal')
    tester.make_demo_image()
    tester.infer_image_by_path()
    # tester.infer_image_by_path('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2011_003942.jpg')
    # tester.infer_image_by_path(image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2010_003358.jpg')

