"""
Code from: https://medium.com/analytics-vidhya/image-flipping-and-mirroring-with-numpy-and-opencv-aecc08558679
"""
import getopt
import sys
from PIL import Image

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class ImageOpsFromScratch(object):
    def __init__(self, image_file, flip_colors=False):
        self.image_file = image_file
        self.flip_colors = flip_colors
    def read_this(self, gray_scale=False):
        file_type = self.image_file.split('.')[1]
        if file_type == "gif":
            cap = cv2.VideoCapture(self.image_file)
            ret, image_src = cap.read()
            cap.release()
        else:
            image_src = cv2.imread(self.image_file)
        if gray_scale:
            image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        else:
            if self.flip_colors:
                image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_src


        return image_rgb
    def mirror_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_mirror = np.fliplr(image_rgb)
        if with_plot:
            self.plot_it(orig_matrix=image_rgb, trans_matrix=image_mirror, head_text='Mirrored', gray_scale=gray_scale)
            return None
        return image_mirror
    def flip_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_flip = np.flipud(image_rgb)
        if with_plot:
            self.plot_it(orig_matrix=image_rgb, trans_matrix=image_flip, head_text='Flipped', gray_scale=gray_scale)
            return None
        return image_flip
    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text(head_text)
        if not gray_scale:
            ax1.imshow(orig_matrix)
            ax2.imshow(trans_matrix)
        else:
            ax1.imshow(orig_matrix, cmap='gray')
            ax2.imshow(trans_matrix, cmap='gray')
        return True


def is_mirrored(path):
    files = os.listdir(path)

    # is the directory already mirrored?
    og_list = []
    mirror_list = []
    for f in files:
        if "_mirror" in f:
            og_f = f.split("_mirror")
            mirror_list.append(f"{og_f[0]}")
        else:
            og_f = f.split(".")
            og_list.append(og_f)

    differences = list(set(og_list) - set(mirror_list)) + list(set(og_list) - set(mirror_list))
    if differences:
        print(f"The following files aren't mirrored:{differences}")
        return False
    else:
        print("Looks like the directory is already mirrored")
        return True



if __name__ == "__main__":
    argv = sys.argv[1:]
    inputpath = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ipath="])
    except getopt.GetoptError:
        print
        'test.py -i <inputpath>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -i <inputpath>'
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputpath = arg

    # inputpath = '/Users/christopherturner/Documents/ExperimentImageSet'
    original_path = f"{inputpath}/original"


    files = os.listdir(original_path)

    # Convert files to jpeg
    # print('the following files have been converted to .jpeg from .gif:')
    # for f in files:
    #     img_path = f"{original_path}/{f}"
    #     f_name = f.split(".")[0]
    #     f_ext = f.split(".")[1]
    #     if f_ext == "gif":
    #         img = Image.open(img_path).convert('RGB')
    #         converted_path = f"{original_path}/{f_name}.jpeg"
    #         img.save(converted_path)
    #         print(f)


    # Mirror the files and save them in the mirrored directory
    mirrored_path = f"{inputpath}/mirrored"
    if not os.path.exists(mirrored_path):
        os.makedirs(mirrored_path)

    files = os.listdir(original_path)
    for f in files:
        if not f.startswith('.'):
            img_path = f"{original_path}/{f}"
            imgs = ImageOpsFromScratch(img_path)
            mirror = imgs.mirror_this(with_plot=False)
            file_name = f.split(".")
            file_ext = file_name[1]
            if file_ext.lower() in ["gif"]:
                file_ext = "jpg"
            mirror_img_path = f"{mirrored_path}/{file_name[0]}.{file_ext}"
            print(f"saved files:{mirror_img_path}")
            cv2.imwrite(mirror_img_path, mirror)
pass