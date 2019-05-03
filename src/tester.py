import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from dataset.pascal_voc import PascalVocDataset
from dataset.deepglobe import DeepGlobeDataset
from utils.preprocess import minmax_normalize
from utils.constants import web_palette_values


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

        self.web_palette = web_palette_values


        print('[Tester] [Init] ...done!')
        print('[Tester] [Init] Tester created.')

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
        pixels = good_mask.load()  # create the pixel map

        for i_index in range(good_mask.size[0]):
            for j_index in range(good_mask.size[1]):
                if pixels[i_index, j_index] == 1:  # BLACK
                    # pixels[i_index, j_index] = 0  # Unknown
                    good_mask.putpixel((i_index, j_index), 0)
                elif pixels[i_index, j_index] == 2:  # GREEN
                    # pixels[i_index, j_index] = 40  # Forest land
                    good_mask.putpixel((i_index, j_index), 40)
                elif pixels[i_index, j_index] == 3:  # YELLOW
                    # pixels[i_index, j_index] = 45  # Agriculture land
                    good_mask.putpixel((i_index, j_index), 45)
                elif pixels[i_index, j_index] == 4:  # BLUE
                    # pixels[i_index, j_index] = 190  # Water
                    good_mask.putpixel((i_index, j_index), 190)
                elif pixels[i_index, j_index] == 5:  # MAGENTA
                    # pixels[i_index, j_index] = 195  # Rangeland
                    good_mask.putpixel((i_index, j_index), 195)
                elif pixels[i_index, j_index] == 6:  # CYAN
                    # pixels[i_index, j_index] = 220  # Urban land
                    good_mask.putpixel((i_index, j_index), 220)
                elif pixels[i_index, j_index] == 7:  # WHITE
                    # pixels[i_index, j_index] = 225  # Barren land
                    good_mask.putpixel((i_index, j_index), 225)
                else:
                    print("[ERROR] Unknown color " + str(pixels[i_index, j_index]))
                    exit(-1)

        good_mask.putpalette(self.web_palette)
        if display:
            good_mask.show()

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
        print('[Tester] [Single test] Opening image ' + image_name + '...')
        # Open and prepare image
        input_img = Image.open('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/JPEGImages/'+image_name+'.jpg')
        if display:
            input_img.show(title='Input raw image')

        label = Image.open('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/SegmentationClass/' + image_name + '.png')
        label = label.convert('P', palette=Image.WEB)
        # web_palette = label.getpalette()  # is equal to self.web_palette
        if display:
            label.show(title='Ground truth')

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
        pixels = good_mask.load()  # create the pixel map

        for i_index in range(good_mask.size[0]):
            for j_index in range(good_mask.size[1]):
                if pixels[i_index, j_index] == 1:  # BLACK
                    # pixels[i_index, j_index] = 0  # Unknown
                    good_mask.putpixel((i_index, j_index), 0)
                elif pixels[i_index, j_index] == 2:  # GREEN
                    # pixels[i_index, j_index] = 40  # Forest land
                    good_mask.putpixel((i_index, j_index), 40)
                elif pixels[i_index, j_index] == 3:  # YELLOW
                    # pixels[i_index, j_index] = 45  # Agriculture land
                    good_mask.putpixel((i_index, j_index), 45)
                elif pixels[i_index, j_index] == 4:  # BLUE
                    # pixels[i_index, j_index] = 190  # Water
                    good_mask.putpixel((i_index, j_index), 190)
                elif pixels[i_index, j_index] == 5:  # MAGENTA
                    # pixels[i_index, j_index] = 195  # Rangeland
                    good_mask.putpixel((i_index, j_index), 195)
                elif pixels[i_index, j_index] == 6:  # CYAN
                    # pixels[i_index, j_index] = 220  # Urban land
                    good_mask.putpixel((i_index, j_index), 220)
                elif pixels[i_index, j_index] == 7:  # WHITE
                    # pixels[i_index, j_index] = 225  # Barren land
                    good_mask.putpixel((i_index, j_index), 225)
                else:
                    print("[ERROR] Unknown color "+str(pixels[i_index, j_index]))
                    exit(-1)

        good_mask.putpalette(self.web_palette)

        if display:
            good_mask.show(title='Prediction')

        overlay = Tester.make_overlay(good_mask, input_img, 100)
        if display:
            overlay.show(title='Overlay bad colors')
        overlay.save(output_name + '_overlay.png')
        print('[Tester] [Single test] Done.')

    @staticmethod
    def make_overlay(pred, img, transparency):
        """
        Build PIL image from input img and mask overlay with given transparency.
        :param pred: mask input
        :param img: img input
        :param transparency: transparency wanted between 0..255
        :return: PIL image result
        """
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
    tester_deepglobe = Tester(model_path='../model/deepglobe_deeplabv3_weights-cityscapes_19-outputs/model.pth', dataset='deepglobe', output_channels=19, split='train', net_type='deeplab', batch_size=1, shuffle=True)
    tester_deepglobe.infer_image_by_path('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/JPEGImages/255876.jpg', display=True, output_name='custom_output')
    # tester_deepglobe.infer_image_by_name(image_name="782103")

    # tester_deepglobe_no_pretrained_7_channels = Tester(model_path='../model/deepglobe_deeplabv3_no_pretrained_7_channels/model.pth', dataset='deepglobe', output_channels=7, split='train', net_type='deeplab', batch_size=1, shuffle=True)
    # tester_deepglobe_no_pretrained_7_channels.make_demo_image()
    # tester_deepglobe_no_pretrained_7_channels.infer_image_by_path('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/JPEGImages/330838.jpg', display=True, output_name='custom_output')

    # tester_pascal = Tester(model_path='../model/pascal_deeplabv3p/model.pth', dataset='pascal')
    # tester_pascal.make_demo_image()
    # tester_pascal.infer_image_by_path(display=True, output_name='custom_output')
    # tester_pascal.infer_image_by_path('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2011_003942.jpg', display=True, output_name='pascal_custom_output')
    # tester_pascal.infer_image_by_path(image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2009_004857.jpg', display=True, output_name='pascal_custom_output_scratch')
    print('[Tester] Tests done.')

    ##################
    # MESSING AROUND #
    ##################

    """
    input_img = Image.open('/home/ubuntu/data/Segmentation/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012/SegmentationClass/96870.png')
    input_img = input_img.convert('P', palette=Image.WEB)
    input_img.thumbnail((50, 50), Image.ANTIALIAS)

    pixels = input_img.load()  # create the pixel map
    for i in range(input_img.size[1]):
        for j in range(input_img.size[0]):
            pixels[i, j] = 225

    input_img.show()
    """

