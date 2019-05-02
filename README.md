# PytorchSegmentation

Forked from https://github.com/nyoki-mtl/pytorch-segmentation

> This repository implements general network for semantic segmentation.
> You can train various networks like DeepLabV3+, PSPNet, UNet, etc., just by writing the config file.

I brought modifications to fix the apex dependency manually. I also adapted the code to take as training input the DeepGlobe2018 land cover dataset containing satellite images. Besides, I added some inference test scripts.

I additionnaly wrote a script to compute the class repartition in the segmentation dataset.
