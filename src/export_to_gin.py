
import numpy as np
from PIL import Image
from pathlib import Path
from utils.constants import color_correspondences, web_palette_values
import os
from shutil import copyfile
import json
import albumentations as albu
from osgeo import gdal
# enforce GDAL exceptions (otherwise functions return None)
gdal.UseExceptions()

dataset_dir= '/home/sfoucher/DEV/pytorch-segmentation/data/deepglobe_as_pascalvoc/VOCdevkit/VOC2012'

out_dir= '/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe'

split = 'val'
def process():
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
                    # For debug
                    # print("[ERROR] Unknown color in label")
                    exit(-1)

        return pil_image
    base_dir = Path(dataset_dir)
    ignore_index = 255
    mode_classif= True
    target_size = (621, 621)
    if mode_classif:
        target_size= (64, 64)
        resizer = albu.Compose([albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
        out_dir = '/home/sfoucher/DEV/geoimagenet/dataset_test/deepglobe_classif'
    
    out_dir = Path(out_dir) / f'{split}'
    if split == 'test':
        valid_ids = base_dir / 'ImageSets' / 'Segmentation' / 'val.txt'
    else:
        valid_ids = base_dir / 'ImageSets' / 'Segmentation' / f'{split}.txt'
    with open(valid_ids, 'r') as f:
        valid_ids = f.readlines()

    lbl_dir = 'SegmentationClass'
    #class_mapping = [('AgriculturalLand', 222), ('BarrenLand', 251), ('ForestLand', 232), ('RangeLand', 228), ('UrbanLand', 199), ('Water', 238)]       
    class_mapping = [('AgriculturalLand', 223), ('BarrenLand', 252), ('ForestLand', 233), ('RangeLand', 229), ('UrbanLand', 201), ('Water', 239)]       
    effectif= {v: 0 for k,v in class_mapping}

    img_ids = valid_ids
    img_paths = [(base_dir / 'JPEGImages' / f'{img_id.strip()}.jpg') for img_id in img_ids]
    lbl_paths = [(base_dir / lbl_dir / f'{img_id.strip()}.png') for img_id in img_ids]
    list_crops= dict()
    list_crops["patches"]= []
    for k, (lbl_path, img_path) in enumerate(zip(lbl_paths, img_paths)):
        print(k)
        inter_img = Image.open(lbl_path)
        img = np.array(Image.open(img_path))
        # Convert the RGB images to the P mode in PIL
        # inter_img = inter_img.convert('P', palette=Image.ADAPTIVE, colors=256) C'EST DE LA MERDE
        inter_img = inter_img.convert('P', palette=Image.WEB)
        lbl = palette_to_indexes(inter_img)
        lbl_np = np.array(inter_img)
        unique, counts = np.unique(lbl_np, return_counts=True)
        n_max= unique[(counts == counts.max())][0]
        if n_max >= len(class_mapping):
            continue
        n_class = class_mapping[n_max][1]
        # lbl_np = (lbl_np == n_max)*255
        lbl_np = (lbl_np == n_max)*(n_max+1) + (lbl == 255) * 255
        lbl_np = lbl_np.astype(np.uint8)
        if mode_classif:
            m = 0
            n = 0
            while n < 0.8 * target_size[0] * target_size[1] and m < 20:
                resized = resizer(image=img, mask=lbl_np)
                img2, lbl = resized['image'], resized['mask']
                n = np.count_nonzero(lbl)
                m += 1
            if m == 20:
                print(f'cant find an image: {n_class}')
                continue
            else:
                effectif[n_class] += 1
                # Image.fromarray(img2).save(Path(out_dir) / f'{Path(img_path).stem}.png')
                lbl_np = lbl
        else:
            copyfile(img_path, Path(out_dir) / Path(img_path).name)
        output_driver = gdal.GetDriverByName("GTiff")
        mask= Image.fromarray(lbl_np)
        mask_name = Path(lbl_path).stem + "_mask.png"
        mask.save(Path(out_dir) / mask_name)
        img_name = f'{Path(img_path).stem}.png'
        filename = "{}.tif".format(Path(img_path).stem)
        output_path = os.path.join(out_dir, filename)
        if os.path.exists(output_path):
            msg = "Output path [{}] already exists but is expected to not exist.".format(output_path)
            print(msg)
            os.remove(output_path)  # gdal raster creation fails if file already exists
        output_dataset = output_driver.Create(output_path, target_size[1], target_size[0], 3, 1)
        for raster_band_idx in range(3):
            output_dataset.GetRasterBand(raster_band_idx + 1).SetNoDataValue(0)
            output_dataset.GetRasterBand(raster_band_idx + 1).WriteArray(img2[:, :, raster_band_idx])
        output_dataset = None  # close output fd
        config_crops= {"crops": [{
                        "type": "raw",
                        "path": filename,
                        "mask": mask_name,
                        "shape": [target_size[0], target_size[1], 3],
                        "data_type": 1,
                        "coordinates": [
                        614354.5,
                        0.5,
                        0.0,
                        5055988.0,
                        0.0,
                        -0.5
                    ]
                }
            ],
            "image": "/data/geoimagenet/images/PLEIADES_RGBN_16/Pleiades_20151010_RGBN_50cm_16bits_AOI_35_Montreal_QC.tif",
            "class": n_class,
            "split": split,
            "feature": "annotation." + str(k)
        }
        list_crops["patches"].append(config_crops)
    with open(Path(out_dir) / 'patches.json', 'w') as outfile:
        json.dump(list_crops, outfile)
    print(effectif)
if __name__ == '__main__':
    process()