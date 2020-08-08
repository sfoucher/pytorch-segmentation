import json 
import torch
from pathlib import Path
from copy import deepcopy
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import time
import copy
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from utils.visualize import VisdomLinePlotter
from datetime import datetime
import os
import sys
import thelper

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 6

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for 
num_epochs = 25

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False
lr=0.001
if feature_extract:
    lr=0.001

global plotter
plotter = VisdomLinePlotter(env_name='Training')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


# keys used across methods to find matching configs, must be unique and non-conflicting with other sample keys
IMAGE_DATA_KEY = "data"     # key used to store temporarily the loaded image data
IMAGE_LABEL_KEY = "label"   # key used to store the class label used by the model
TEST_DATASET_KEY = "dataset"

DATASET_FILES_KEY = "files"             # list of all files in the dataset batch
DATASET_DATA_KEY = "data"               # dict of data below
DATASET_DATA_TAXO_KEY = "taxonomy"
DATASET_DATA_MAPPING_KEY = "taxonomy_model_map"     # taxonomy ID -> model labels
DATASET_DATA_ORDERING_KEY = "model_class_order"     # model output classes (same indices)
DATASET_DATA_PATCH_KEY = "patches"
DATASET_DATA_PATCH_CLASS_KEY = "class"       # class id associated to the patch
DATASET_DATA_PATCH_SPLIT_KEY = "split"       # group train/test of the patch
DATASET_DATA_PATCH_CROPS_KEY = "crops"       # extra data such as coordinates
DATASET_DATA_PATCH_IMAGE_KEY = "image"       # original image path that was used to generate the patch
DATASET_DATA_PATCH_MASK_KEY = 'path_mask'    # original mask path that was used to generate the patch
DATASET_DATA_PATCH_FEATURE_KEY = "feature"   # annotation reference id
DATASET_BACKGROUND_ID = 999                  # background class id

DATASET_ROOT= Path('/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif')
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(WORKSPACE_DIR))

def process():

    timestamp = datetime.timestamp(datetime.now())
    print("timestamp =", datetime.fromtimestamp(timestamp))
    output_dir = Path(os.path.join(ROOT_DIR, f'model/{model_name}_{datetime.fromtimestamp(timestamp)}') )
    output_dir.mkdir(exist_ok=True)

    # Opening JSON file 
    with open('/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif/train/meta.json') as json_file: 
        meta = json.load(json_file) 
    dataset= {'path': DATASET_ROOT / 'train', DATASET_DATA_KEY : meta['patches']}
    with open('/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif/val/meta.json') as json_file: 
        meta = json.load(json_file)
    dataset_val= {'path': DATASET_ROOT / 'val', DATASET_DATA_KEY : meta['patches']}

    class_mapping = [('AgriculturalLand', 222), ('BarrenLand', 251), ('ForestLand', 232), ('RangeLand', 228),
                        ('UrbanLand', 199), ('Water', 238)]
    #model_class_map= {v: k k, v in class_mapping} 
    model_class_map = {223 : 'AgriculturalLand', 252 : 'BarrenLand', 233 : 'ForestLand', 229 : 'RangeLand', 200 : 'UrbanLand', 239 : 'Water',}
    model_class_map = {223 : 0, 252 : 1, 233 : 2, 229 : 3, 200 : 4, 239 : 5,}
    class GINParser(torch.utils.data.Dataset):
        def __init__(self, dataset=None, transforms=None):
            if not (isinstance(dataset, dict) and len(dataset)):
                raise ValueError("Expected dataset parameters as configuration input.")
            # thelper.data.Dataset.__init__(self, transforms=transforms, deepcopy=False)
            self.root = dataset["path"]
            self.transforms = transforms
            # keys matching dataset config for easy loading and referencing to same fields
            self.image_key = IMAGE_DATA_KEY     # key employed by loader to extract image data (pixel values)
            self.label_key = IMAGE_LABEL_KEY    # class id from API mapped to match model task
            self.path_key = "path"              # actual file path of the patch
            self.idx_key = "index"              # increment for __getitem__
            self.mask_key = 'mask'        # actual mask path of the patch
            self.meta_keys = [self.path_key, self.idx_key, DATASET_DATA_PATCH_CROPS_KEY,
                                DATASET_DATA_PATCH_IMAGE_KEY, DATASET_DATA_PATCH_FEATURE_KEY]
            # model_class_map = dataset[DATASET_DATA_KEY][DATASET_DATA_MAPPING_KEY]
            sample_class_ids = set()
            samples = []
            for patch_info in dataset['data']:
                # convert the dataset class ID into the model class ID using mapping, drop sample if not found
                class_name = model_class_map.get(patch_info.get('class'))
                if class_name is not None:
                    sample_class_ids.add(class_name)
                    samples.append(deepcopy(patch_info))
                    samples[-1][self.path_key] = self.root / patch_info['crops'][0]['path']
                    samples[-1][self.label_key] = class_name

            if not len(sample_class_ids):
                raise ValueError("No patch/class could be retrieved from batch loading for specific model task.")
            self.samples = samples
            self.sample_class_ids = sample_class_ids

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sample = self.samples[idx]
            img_name = sample[self.path_key]
            # image = cv.imread(img_name._str)
            image = Image.open(img_name)
            assert image is not None, "could not load image '%s' via opencv" % sample[self.path_key]
            #tensor_to_PIL = torchvision.transforms.ToPILImage(mode='RGB')
            #image = tensor_to_PIL(image)
            sample = {
                self.image_key: image,
                self.path_key: sample[self.path_key],
                self.label_key: sample[self.label_key],
                self.idx_key: idx
            }
            
            if self.transforms:
                image = self.transforms(image)

            return image, sample[self.label_key]

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets= dict()
    dataloaders_dict= dict()
    image_datasets['train'] = GINParser(dataset = dataset, transforms = data_transforms['train'])
    image_datasets['val'] = GINParser(dataset = dataset_val, transforms = data_transforms['val'])
    dataloaders_dict['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)

    # Print the model we just instantiated
    print(model_ft)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr = lr, momentum=0.9)
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
        
        optimizer_ft = optim.SGD(params_to_update, lr = lr, momentum=0.9)
    else:
        n_param= 0
        name_block = dict()
        lr_blcok = dict()
        for l, (name,param) in enumerate(model_ft.named_parameters()):
            n_param += 1
        for l, (name,param) in enumerate(model_ft.named_parameters()):
            blcok = name.split('.')[0]
            name_block[l] = blcok
            lr_blcok[blcok] = lr * 10**(2*(l/n_param-1))
            
        params_to_update = []
        #name_block = set(name_block)
        for l, (name,param) in enumerate(model_ft.named_parameters()):
            if l < int(n_param)-8:
                param.requires_grad = False
            else:
                if 'fc.' not in name:
                    params_to_update.append({
                        "params": param,
                        "lr": lr_blcok[name_block[l]],
                    })
                else:
                    params_to_update.append({
                        "params": param,
                        "lr": lr,
                    })
            if param.requires_grad == True:
                print("\t",name)
        
        optimizer_ft = optim.SGD(params_to_update, momentum=0.9,weight_decay=0.001)
    
    

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()


    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        ma_loss = 0.0
        ma_iou = 0.0
        i_iter = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in dataloaders.keys():
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                with tqdm(dataloaders[phase]) as _tqdm:
                    for i, (inputs, labels) in enumerate(_tqdm):
                        #for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if is_inception and phase == 'train':
                                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            acc = np.logical_and(preds.cpu().numpy(), labels.cpu().numpy()).sum() / len(labels)
                            _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', acc=f'{acc*100:.3f}'))
                            
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                                ma_loss= 0.01*loss.item() +  0.99 * ma_loss
                                ma_iou= 0.01*acc +  0.99 * ma_iou
                                plotter.plot('loss', 'train', 'iteration Loss', i_iter, loss.item()) # y-axis, name serie, name chart
                                plotter.plot('acc', 'train', 'iteration acc', i_iter, acc)
                                plotter.plot('loss', 'ma_loss', 'iteration Loss', i_iter, ma_loss)
                                plotter.plot('acc', 'ma_acc', 'iteration acc', i_iter, ma_iou)
                                i_iter += 1
                            else:
                                i_iter = i_iter
                                
                            

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double().item() / len(dataloaders[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'train':
                    plotter.plot('loss-epoch', 'train', 'epoch Loss', epoch, epoch_loss)
                    plotter.plot('acc-epoch', 'train', 'epoch acc', epoch, epoch_acc)
                else:
                    plotter.plot('loss-epoch', 'valid', 'epoch Loss', epoch, epoch_loss)
                    plotter.plot('acc-epoch', 'valid', 'epoch acc', epoch, epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    print('Best Epoch!')
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
                    torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()
            # scheduler.step()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history


    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


if __name__ == '__main__':
    process()