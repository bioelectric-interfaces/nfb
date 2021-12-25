"""
Script to generate free viewing task for each participant
This script randomly selects half the test images for the pre- and the other half for the post- task
It ensures that images are evenly distributed between the original and mirror version
Every image is only shown once per participant either in its original or mirrored form
"""
import os
import random

class ImageListGenerator:
    def __init__(self, root_path="../defaults"):
        # get the number of images in the directory
        self.root_path = root_path
        original_path = f"{root_path}/original"
        mirrored_path = f"{root_path}/mirrored"
        self.directories = [original_path, mirrored_path]
        self.files = os.listdir(original_path)

        # Remove .DS_Store if it exists
        if ".DS_Store" in self.files:
            self.files.remove(".DS_Store")

    def get_pre_post_images(self):
        selected_images = {}
        dir = 0
        n = len(self.files)
        for x in range(0, n):
            # for each image, randomly select between /original and /mirrored directory
            dir = random.randint(0, 1)
            path = self.directories[dir]

            # then, randomly choose an image and update the dictionary with image name, and absolute path
            len_imgs = len(self.files)
            img_idx = random.randint(0, len_imgs-1)
            print(f"x: {x}, img_idx: {img_idx}, len_imgs: {len_imgs}")
            img = self.files[img_idx]
            del self.files[img_idx]
            selected_images[img] = {"path": f"{path}/{img}"}

        # split the now random dictionray in half - one for pre, and one for post
        pre_task_images = {}
        post_task_images = {}
        for i, k in enumerate(selected_images.items()):
            if i > len(selected_images)/2:
                pre_task_images[k[0]] = k[1]
            else:
                post_task_images[k[0]] = k[1]

        return pre_task_images, post_task_images
