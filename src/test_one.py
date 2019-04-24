import numpy as np
import matplotlib
import torch
import cv2
from PIL import Image
from models.net import SPPNet


class Tester:
    def __init__(self):
        matplotlib.use('Agg')

    @staticmethod
    def infer_image_by_path(image_path='/home/ubuntu/data/Segmentation/pytorch-segmentation/test1.jpg',
                            model_path='../model/cityscapes_deeplab_v3_plus/model.pth'):
        print('Loading model '+model_path+'...')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SPPNet(output_channels=19).to(device)
        param = torch.load(model_path)
        model.load_state_dict(param)
        del param

        print('...done!')

        # Notify layers that we are in eval mode (for batchnorm, dropout)
        model.eval()
        # Deactivate autograd engine to reduce memory usage (no need backprop when inferring)
        with torch.no_grad():
            print('Preparing image '+image_path+'...')
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Transpose image to fit torch tensor format
            img = img.transpose(2, 0, 1)
            # Conversion to torch tensor
            img_tensor = torch.tensor([img])
            # Conversion to float to fit torch model
            img_tensor = img_tensor.float()

            print('Inferring...')

            # Send to CPU or GPU depending on the hardware found
            img_tensor = img_tensor.to(device)

            # Generate predictions
            preds = model.tta(img_tensor, net_type='deeplab')
            # Not sure what this does in initial code, probably helps extracting a single class
            preds = preds.argmax(dim=1)

            # Convert back to nparray procssable as an image
            preds_np = preds.detach().cpu().numpy()
            preds_np = preds_np[0]

            # Probably need to transform preds to a png now !
            print('Generating mask...')

            pred_pil = Image.fromarray(preds_np.astype(np.uint8))
            pred_pil.show()

            print('Done.')


if __name__ == '__main__':
    tester = Tester()
    # tester.infer_image_by_path()
    tester.infer_image_by_path(image_path='../data/pascal_voc_2012/VOCdevkit/VOC2012/JPEGImages/2007_000549.jpg',
                               model_path='../model/pascal_deeplabv3p_with_pretrained/model.pth')
