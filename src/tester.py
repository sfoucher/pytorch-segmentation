import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from models.net import SPPNet, EncoderDecoderNet
from dataset.cityscapes import CityscapesDataset
from dataset.pascal_voc import PascalVocDataset
from dataset.deepglobe import DeepGlobeDataset
from utils.preprocess import minmax_normalize

import matplotlib.pyplot as plt

import copy
import sys, os
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(WORKSPACE_DIR))
sys.path.insert(0, ROOT_DIR)

class Tester:
    def __init__(self, model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs/model.pth', dataset='deepglobe',
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
        model_path= '/home/sfoucher/DEV/pytorch-segmentation/model/my_pascal_unet_res18_scse/model.pth'
        dataset_dir = '/home/sfoucher/DEV/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012'
        
        output_channels= 8
        net_type= 'unet'
        print('[Tester] [Init] Initializing tester...')
        self.dataset = dataset
        self.model_path = model_path

        # Load model
        print('[Tester] [Init] Loading model ' + model_path + ' with ' + str(output_channels) + ' output channels...')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if net_type == 'unet':
            self.model = EncoderDecoderNet(output_channels= 8, enc_type='resnet18', dec_type='unet_scse',
                 num_filters=8)
        else:
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
            self.valid_dataset = DeepGlobeDataset(base_dir = dataset_dir, target_size=(64, 64), split=split, net_type=net_type)
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

    def infer_image_by_path(self,
                            image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/test1.jpg',
                            output_name='single_test_output',
                            display=False):
        """
        Opens image from fs and passes it through the loaded network, then displays and saves the result.
        :param output_name: Output image name
        :param display: Display images in windows or not
        :param image_path: Path of input images
        :return: null
        """
        if not self.dataset == 'deepglobe':
            print('[ERROR] Inference script only available for the Deepglobe dataset.')
            exit(-1)

        print('[Tester] [Single test] Opening image '+image_path+'...')
        # Open and prepare image
        input_img = Image.open(image_path)
        if display:
            input_img.show()

        custom_img = np.array(input_img)
        custom_img = minmax_normalize(custom_img, norm_range=(-1, 1))
        custom_img = custom_img.transpose(2, 0, 1)
        custom_img = torch.FloatTensor([custom_img])

        print('[Tester] [Single test] Inferring image...')
        self.model.eval().to(self.device)
        with torch.no_grad():
            # Send to GPU, infer and collect
            custom_img = custom_img.to(self.device)
            #preds = self.model.tta(custom_img, net_type='deeplab')
            preds = self.model.tta(custom_img, net_type='unet')
            preds = preds.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()

        print('[Tester] [Single test] Processing result...')

        good_preds = preds_np[0]
        good_mask = Image.fromarray(good_preds.astype('uint8'), 'P')

        # Transform mask to set good indexes and palette
        good_mask = DeepGlobeDataset.index_to_palette(good_mask)

        if display:
            good_mask.show()
        good_mask.save(output_name+'_prediction.png')

        overlay = Tester.make_overlay(good_mask, input_img, 100)
        if display:
            overlay.show()
        overlay.save(output_name+'_overlay.png')
        print('[Tester] [Single test] Done.')

    def infer_image_by_name(self,
                            image_name='255876',
                            output_name='single_test_output',
                            display=True):
        """
        Opens image from fs and passes it through the loaded network, then displays and saves the result.
        :param output_name: Output image name
        :param display: Display images in windows or not
        :param image_path: Path of input images
        :return: null
        """
        if not self.dataset == 'deepglobe':
            print('[ERROR] Inference script only available for the Deepglobe dataset.')
            exit(-1)

        print('[Tester] [Single test] Opening image ' + image_name + '...')
        # Open and prepare image
        input_img = Image.open('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/JPEGImages/'+image_name+'.jpg')
        label = Image.open('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/SegmentationClass/' + image_name + '.png')
        label_raw = copy.deepcopy(label)
        overlay_ground_truth = Tester.make_overlay(label_raw, input_img, 100)
        label = label.convert('P', palette=Image.WEB)

        if display:
            input_img.show(title='Input raw image')
            label.show(title='Ground truth')
            overlay_ground_truth.show(title='Overlay_ground_truth')

        custom_img = np.array(input_img)
        custom_img = minmax_normalize(custom_img, norm_range=(-1, 1))
        custom_img = custom_img.transpose(2, 0, 1)
        custom_img = torch.FloatTensor([custom_img])

        print('[Tester] [Single test] Inferring image...')
        self.model.eval()
        with torch.no_grad():
            # Send to GPU, infer and collect
            custom_img = custom_img.to(self.device)
            preds = self.model.tta(custom_img, net_type='deeplab')
            preds = preds.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()

        print('[Tester] [Single test] Processing result...')

        good_preds = preds_np[0]
        good_mask = Image.fromarray(good_preds.astype('uint8'), 'P')

        # Transform mask to set good indexes and palette
        good_mask = DeepGlobeDataset.index_to_palette(good_mask)

        overlay = Tester.make_overlay(good_mask, input_img, 100)
        if display:
            good_mask.show(title='Prediction')
            overlay.show(title='Overlay')

        good_mask.save(output_name + '_prediction.png')
        overlay.save(output_name + '_overlay.png')
        overlay_ground_truth.save(output_name + '_overlay_truth.png')

        print('[Tester] [Single test] Done.')

    @staticmethod
    def make_overlay(pred_in, img_in, transparency):
        """
        Build PIL image from input img and mask overlay with given transparency.
        :param pred_in: mask input
        :param img_in: img input
        :param transparency: transparency wanted between 0..255
        :return: PIL image result
        """
        pred = copy.deepcopy(pred_in)
        img = copy.deepcopy(img_in)
        print('[Tester] [Overlay] Building overlay...')
        if transparency < 0 or transparency > 255:
            print('ERROR : Transparency should be in range 0..255.')
            exit(-1)
        # Make preds semi_transparent
        pred = pred.convert('RGBA')
        data = pred.getdata()  # you'll get a list of tuples
        new_data = []
        for a in data:
            a = a[:3]  # you'll get your tuple shorten to RGB
            a = a + (transparency,)  # change the 100 to any transparency number you like between (0,255)
            new_data.append(a)
        pred.putdata(new_data)  # you'll get your new img ready

        # Paste translucid preds on input image
        img.paste(pred, (0, 0), pred)
        print('[Tester] [Overlay] Done.')
        return img


if __name__ == '__main__':
    print('[Tester] Launching tests.')
    tester_deepglobe = Tester(model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs_small-patches_dynamic/model.pth', dataset='deepglobe', output_channels=19, split='valid', net_type='deeplab', batch_size=1, shuffle=True)
    tester_deepglobe.infer_image_by_path('/home/sfoucher/DEV/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/JPEGImages/115444_4-2.jpg', display=True, output_name='custom_output')
    # tester_deepglobe.infer_image_by_name(image_name="255876", display=False)
    # tester_deepglobe.make_demo_image()

    # tester_pascal = Tester(model_path='../model/pascal_deeplabv3p_with_pretrained/model.pth', dataset='pascal')
    # tester_pascal.make_demo_image()
    # tester_pascal.infer_image_by_path(display=True, output_name='custom_output')
    # tester_pascal.infer_image_by_path('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2011_003942.jpg', display=True, output_name='pascal_custom_output')
    # tester_pascal.infer_image_by_path(image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2009_004857.jpg', display=True, output_name='pascal_custom_output_scratch')
    print('[Tester] Tests done.')

