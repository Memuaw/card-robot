import numpy as np
import cv2
import os
import random
from glob import glob 
import imgaug as ia
from imgaug import augmenters as iaa
from functions import generate_non_overlapping_transforms, overlay_image, randomly_adjust_hue_brightnesses, apply_transform, transform_points, compute_bounding_box

class Scene():
    def __init__(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3=None, class3=None, hulla3=None, hullb3=None):
        if img3 is not None:
            self.create_3_card_scene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3, hullb3)
        else:
            self.scene = self.create_2_card_scene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2)
    
    def create_2_card_scene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2):
        # Declare our variables
        self.bg = bg
        self.hulla1 = hulla1
        self.hullb1 = hullb1
        self.hulla2 = hulla2
        self.hullb2 = hullb2
        self.img1 = img1
        self.img2 = img2
        self.class1 = class1
        self.class2 = class2

        # Adjust brightness and hue randomly
        self.img1, self.img2 = randomly_adjust_hue_brightnesses(self.img1, self.img2)

        # Generate our random transforms
        self.scale1, self.rotate1, self.translate_percent1, self.scale2, self.rotate2, self.translate_percent2 = generate_non_overlapping_transforms(img1, img2, self.bg.shape[:2])

        # Apply our transforms to our images
        self.img1, M1 = apply_transform(self.img1, self.scale1, self.rotate1, self.translate_percent1, self.bg.shape[:2])
        self.img2, M2 = apply_transform(self.img2, self.scale2, self.rotate2, self.translate_percent2, self.bg.shape[:2])

        # Paste the two cards on to the background
        self.final, flip1 = overlay_image(self.bg, self.img1, (0, 0))
        self.final, flip2 = overlay_image(self.final, self.img2, (0, 0))

        # Apply our tranforms to our hull points
        self.transformed_hull_a1, M = transform_points(self.hulla1, M1)
        self.transformed_hull_b1, M = transform_points(self.hullb1, M)
        self.transformed_hull_a2, M = transform_points(self.hulla2, M2)
        self.transformed_hull_b2, M = transform_points(self.hullb2, M)
        
        # Convert the transformed hull points into bounding boxes
        self.bboxa1 = compute_bounding_box(self.transformed_hull_a1, flip1)
        self.bboxa2 = compute_bounding_box(self.transformed_hull_a2, flip2)
        self.bboxb1 = compute_bounding_box(self.transformed_hull_b1, flip1)
        self.bboxb2 = compute_bounding_box(self.transformed_hull_b2, flip2)
        self.bbox_list = [self.bboxa1, self.bboxa2, self.bboxb1, self.bboxb2]

    def create_3_card_scene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3, hullb3):
        pass