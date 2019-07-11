from dataset.deepglobe import DeepGlobeDataset
from PIL import Image
from utils.constants import color_correspondences
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    def __init__(self):
        self.dataset_train = DeepGlobeDataset(net_type='deeplab', split='train')
        self.dataset_valid = DeepGlobeDataset(net_type='deeplab', split='valid')

        self.number_of_pixels_dict = {45: 1033012, 190: 59065, 220: 198381, 195: 148438, 0: 965, 225: 146374, 40: 193758}

        # Valid : {45: 126090, 195: 21857, 190: 7402, 220: 18650, 225: 23158, 40: 30291, 0: 45}
        # Train : {45: 1033012, 190: 59065, 220: 198381, 195: 148438, 0: 965, 225: 146374, 40: 193758}

    def get_nb_train_images(self):
        return self.dataset_train.__len__()

    def get_nb_valid_images(self):
        return self.dataset_valid.__len__()

    def make_pie_chart(self):
        colors = []
        ratios = []
        total_nb_pixels = 0
        for val in self.number_of_pixels_dict:
            if val == color_correspondences['black']['web-palette']:
                colors.append(color_correspondences['black']['description'])
            elif val == color_correspondences['green']['web-palette']:
                colors.append(color_correspondences['green']['description'])
            elif val == color_correspondences['yellow']['web-palette']:
                colors.append(color_correspondences['yellow']['description'])
            elif val == color_correspondences['blue']['web-palette']:
                colors.append(color_correspondences['blue']['description'])
            elif val == color_correspondences['cyan']['web-palette']:
                colors.append(color_correspondences['cyan']['description'])
            elif val == color_correspondences['magenta']['web-palette']:
                colors.append(color_correspondences['magenta']['description'])
            elif val == color_correspondences['white']['web-palette']:
                colors.append(color_correspondences['white']['description'])
            else:
                print("[ERROR] Unknown color "+str(val))
                exit(-1)
            total_nb_pixels += self.number_of_pixels_dict[val]
        for val in self.number_of_pixels_dict:
            ratios.append(float(self.number_of_pixels_dict[val]) / float(total_nb_pixels))

        def func(pct):
            return "{:.1f}%".format(pct)

        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        wedges, texts, autotexts = ax.pie(ratios, autopct=lambda pct: func(pct),
                                          textprops=dict(color="w"))

        ax.legend(wedges, colors,
                  title="Legend",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")

        ax.set_title("Segmentation class repartition")

        plt.savefig('pie.png')
        plt.show()

    def compute_mask_colors_ratio(self):
        label_values = []
        self.number_of_pixels_dict = {}
        img_index = 0
        for input_img_path in self.dataset_train.lbl_paths:  # !!!
            img_index += 1
            print('[Dataset analyzer] [Colors ratio] Analyzing image '+str(img_index)+' of '+str(self.get_nb_train_images())+' ('+str(input_img_path)+')')
            input_img = Image.open(input_img_path)
            # Reduce mask size to speed up the process
            input_img = input_img.convert('P', palette=Image.WEB)
            input_img.thumbnail((50, 50), Image.ANTIALIAS)
            for val in input_img.getdata():
                if val not in label_values:
                    self.number_of_pixels_dict[val] = 1
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
    dataset.compute_mask_colors_ratio()
    dataset.make_pie_chart()
