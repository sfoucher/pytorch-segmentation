from dataset.deepglobe import DeepGlobeDataset
from PIL import Image


class DatasetAnalyzer:
    def __init__(self):
        self.dataset_train = DeepGlobeDataset(net_type='deeplab', split='train')
        self.dataset_valid = DeepGlobeDataset(net_type='deeplab', split='valid')

        # Valid : {0: 92462, 1: 72141, 2: 37320, 3: 21104, 4: 4301, 5: 166}
        # Train : {3: 131890, 1: 631000, 0: 734071, 4: 32710, 5: 5585, 2: 244566, 6: 171}
        self.number_of_pixels_dict = {3: 131890, 1: 631000, 0: 734071, 4: 32710, 5: 5585, 2: 244566, 6: 171}

    def get_nb_train_images(self):
        return self.dataset_train.__len__()

    def get_nb_valid_images(self):
        return self.dataset_valid.__len__()

    def print_mask_colors_ratio(self):
        label_values = []
        self.number_of_pixels_dict = {}
        img_index = 0
        for input_img_path in self.dataset_train.lbl_paths:
            img_index += 1
            print('[Dataset analyzer] [Colors ratio] Analyzing image '+str(img_index)+' of '+str(self.get_nb_train_images())+' ('+str(input_img_path)+')')
            input_img = Image.open(input_img_path)
            # Reduce mask size to speed up the process
            input_img = input_img.convert('P', palette=Image.ADAPTIVE, colors=256)
            input_img.thumbnail((50, 50), Image.ANTIALIAS)
            for val in input_img.getdata():
                if val not in label_values:
                    self.number_of_pixels_dict[val] = 0
                    label_values.append(val)
                else:
                    self.number_of_pixels_dict[val] += 1
        print('[Dataset analyzer] [Colors ratio] Label values : ' + str(label_values))
        print('[Dataset analyzer] [Colors ratio] Number of pixels : ' + str(self.number_of_pixels_dict))
        total_nb_pixels = 0
        for val in self.number_of_pixels_dict:
            total_nb_pixels += self.number_of_pixels_dict[val]
        for val in self.number_of_pixels_dict:
            if total_nb_pixels != 0:
                print('[Dataset analyzer] [Colors ratio] Ratio of '+str(val)+' : '+str(float(self.number_of_pixels_dict[val])/float(total_nb_pixels))+"%.")


if __name__ == '__main__':
    dataset = DatasetAnalyzer()
    print('[Dataset analyzer] Number of train images : '+str(dataset.get_nb_train_images()))
    print('[Dataset analyzer] Number of validation images : '+str(dataset.get_nb_valid_images()))
    dataset.print_mask_colors_ratio()
