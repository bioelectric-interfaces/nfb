# import the necessary packages

import cv2
import os

# load the input image
image_dir = "/Users/christopherturner/Documents/ExperimentImageSet/bagherzadeh/image_stimuli/0/original"
files = os.listdir(image_dir)

for f in files:
    if not f.startswith('.'):
        ext = f.split('.')
        if ext[1] in ['jpeg', 'tif', 'jpg']:
            print(f)
            image_path = os.path.join(image_dir, f)
            image = cv2.imread(image_path)
            # initialize OpenCV's static saliency spectral residual detector and
            # compute the saliency map
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = saliency.computeSaliency(image)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            cv2.imshow("Image", image)
            cv2.imshow("Output", saliencyMap)
            saliencyMap_left = saliencyMap[:, 0:int(saliencyMap.shape[1]/2) - 1]
            # cv2.imshow("Output_Left", saliencyMap_left)
            #
            saliencyMap_right = saliencyMap[:, int(saliencyMap.shape[1]/2) : -1]
            # cv2.imshow("Output_Right", saliencyMap_right)

            saliency_mean = saliencyMap.mean()
            saliency_std = saliencyMap.std()
            saliency_max = saliencyMap.max()

            saliency_left_mean = saliencyMap_left.mean()
            saliency_right_mean = saliencyMap_right.mean()

            # Image is OK if has evenly distributed saliency # TODO: how to do this computationally?
            # Image is not lateralised if left and right saliency are within one std of eachother TODO: check this exactly
            # Image is also useful if the max saliency is not above one std of the overall mean

            if saliency_max <= saliency_mean + saliency_std:
                print(f"IMAGE OK: {image_path}")
            cv2.waitKey(0)
            
            # initialize OpenCV's static fine grained saliency detector and
            # compute the saliency map
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency.computeSaliency(image)
            # if we would like a *binary* map that we could process for contours,
            # compute convex hull's, extract bounding boxes, etc., we can
            # additionally threshold the saliency map
            threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # show the images
            cv2.imshow("Image", image)
            cv2.imshow("Output", saliencyMap)
            cv2.imshow("Thresh", threshMap)
            cv2.waitKey(0)
