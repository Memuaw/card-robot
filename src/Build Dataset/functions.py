import cv2
import numpy as np
import random
from imgaug import augmenters as iaa
import glob
from convex_hull_gen import convex_hull_list
import math

def get_random_background():
    categories = ['blotchy', 'braided', 'crystalline', 'flecked', 'grooved', 'lined', 'woven']
    category = random.choice(categories)
    background = cv2.imread(random.choice(glob.glob(f"input/Card Detector/Build Dataset/Background Images/{category}/*.jpg")))
    background = cv2.resize(background, (720, 720))

    return background

def get_random_cards():
    card_images = glob.glob("input/Card Detector/Build Dataset/Card Images/*.jpg")

    #index = random.randint(0, 51)
    index = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    image = cv2.imread(card_images[index])

    original_height, original_width = image.shape[:2]
    target_height, target_width = 320, 176

    h_times = original_height / target_height
    w_times = original_width / target_width

    image = cv2.resize(image, (target_width, target_height))

    name = get_name(index)

    hull_a = convex_hull_list[index][0]
    hull_b = convex_hull_list[index][1]
    
    hull_a_resized = [(x / w_times, y / h_times) for [[x, y]] in hull_a]
    hull_b_resized = [(x / w_times, y / h_times) for [[x, y]] in hull_b]

    return image, name, hull_a_resized, hull_b_resized, index

def randomly_adjust_hue_brightnesses(card1, card2):
    hue = random.randint(-15, 15)
    brightness = random.randint(-15, 22)
    augment = iaa.Sequential([
        iaa.AddToBrightness(brightness),
        iaa.AddToHue(hue)
    ])

    filtered_card1 = augment(image=card1)
    filtered_card2 = augment(image=card2)

    return filtered_card1, filtered_card2

def apply_transform(card, scale, rotate, translate_percent, bg_size):
    h, w = card.shape[:2]

    # Create the affine transformation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate, scale)
    M[0, 2] += translate_percent[0] * bg_size[1]
    M[1, 2] += translate_percent[1] * bg_size[0]

    # Apply the affine transformation
    transformed_card = cv2.warpAffine(card, M, (bg_size[1], bg_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return transformed_card, M

def transform_points(hull_points, M):
    transformed_points = []
    org_M = M
    
    for point in hull_points:
        if point[0] > 15 and point[0] < 100:
            pt = np.array([point[0] + 7, point[1] + 14, 1])
        elif point[0] > 100 and point[0] < 160:
            pt = np.array([point[0] - 7, point[1] - 14, 1])
        elif point[0] > 160:
            pt = np.array([point[0] + 2.5, point[1] + 5, 1])
        else:
            pt = np.array([point[0], point[1], 1])

        transformed_point = np.dot(M, pt)
        transformed_points.append(transformed_point)
    
    M = org_M

    return transformed_points, M

def compute_bounding_box(points, flip):
    pnts = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(pnts)
    box = cv2.boxPoints(rect)
    box_int = np.int0(box)

    if flip:
        image_center = np.array([360, 360])

        flip_transform = np.array([[-1, 0], [0, -1]])

        box_int = np.dot(box_int - image_center, flip_transform) + image_center
    
    return box_int

def is_within_bounds(card, bg_size, translate_percent):
    card_h, card_w = card.shape[:2]
    bg_h, bg_w = bg_size
    
    x_offset = int(translate_percent[0] * bg_w)
    y_offset = int(translate_percent[1] * bg_h)
    
    if x_offset < 0 or y_offset < 0:
        return False
    if x_offset + card_w > bg_w or y_offset + card_h > bg_h:
        return False
    return True

def generate_non_overlapping_transforms(card1, card2, bg_size):
    while True:
        scale1 = random.uniform(0.65, 0.8)
        rotate1 = random.uniform(-90, 90)
        translate_percent1 = [random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)]
        
        scale2 = scale1 - random.uniform(-0.2, 0.05)
        rotate2 = random.uniform(-90, 90)
        translate_percent2 = [random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)]
        
        if (is_within_bounds(card1, bg_size, translate_percent1) and
            is_within_bounds(card2, bg_size, translate_percent2)):
            break
            
    return scale1, rotate1, translate_percent1, scale2, rotate2, translate_percent2

def get_name(card):
    name = ''

    if card == 0:
        name = '6D'
    if card == 1:
        name = '6S'
    if card == 2:
        name = '5H'
    if card == 3:
        name = '9H'
    if card == 4:
        name = '7C'
    if card == 5:
        name = 'AS'
    if card == 6:
        name = 'AD'
    if card == 7:
        name = '6C'
    if card == 8:
        name = '8H'
    if card == 9:
        name = '7D'
    if card == 10:
        name = '4H'
    if card == 11:
        name = '7S'
    if card == 12:
        name = 'AC'
    if card == 13:
        name = 'QH'
    if card == 14:
        name = 'JC'
    if card == 15:
        name = '10D'
    if card == 16:
        name = '10S'
    if card == 17:
        name = 'KS'
    if card == 18:
        name = 'KD'
    if card == 19:
        name = '3H'
    if card == 20:
        name = '2H'
    if card == 21:
        name = 'JS'
    if card == 22:
        name = 'JD'
    if card == 23:
        name = 'KC'
    if card == 24:
        name = '10C'
    if card == 25:
        name = '3C'
    if card == 26:
        name = '2S'
    if card == 27:
        name = '2D'
    if card == 28:
        name = 'JH'
    if card == 29:
        name = 'QC'
    if card == 30:
        name = '10H'
    if card == 31:
        name = 'KH'
    if card == 32:
        name = '3S'
    if card == 33:
        name = '3D'
    if card == 34:
        name = 'QS'
    if card == 35:
        name = 'QD'
    if card == 36:
        name = '2C'
    if card == 37:
        name = '7H'
    if card == 38:
        name = '4S'
    if card == 39:
        name = '4D'
    if card == 40:
        name = '9C'
    if card == 41:
        name = '5C'
    if card == 42:
        name = '8S'
    if card == 43:
        name = '8D'
    if card == 44:
        name = '9S'
    if card == 45:
        name = '9D'
    if card == 46:
        name = '4C'
    if card == 47:
        name = 'AH'
    if card == 48:
        name = '8C'
    if card == 49:
        name = '5S'
    if card == 50:
        name = '6H'
    if card == 51:
        name = '5D'

    return name

def overlay_image(background, img, top_left=None):
    flip_code = random.choice([0, 1])
    flip = False
    if flip_code == 0:
        flip = True
        img = cv2.flip(img, -1)

    h, w = img.shape[:2]
    y, x = top_left

    # Take ROI from background based on img size
    roi = background[y:y+h, x:x+w]

    # Create a mask of the img and create its inverse mask
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of img in ROI
    background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of img from img
    img_fg = cv2.bitwise_and(img, img, mask=mask)

    # Put img in ROI and modify the background
    dst = cv2.add(background_bg, img_fg)
    background[y:y+h, x:x+w] = dst

    return background, flip

def get_valid_bboxes(bbox_list, rotate1, rotate2, names): 
    min_x1a = bbox_list[0][0][0]
    max_x1a = bbox_list[0][2][0]
    if min_x1a > max_x1a:
        min_x1a, max_x1a = max_x1a, min_x1a
    min_y1a = bbox_list[0][0][1]
    max_y1a = bbox_list[0][2][1]
    if min_y1a > max_y1a:
        min_y1a, max_y1a = max_y1a, min_y1a

    min_x1b = bbox_list[2][0][0]
    max_x1b = bbox_list[2][2][0]
    if min_x1b > max_x1b:
        min_x1b, max_x1b = max_x1b, min_x1b
    min_y1b = bbox_list[2][0][1]
    max_y1b = bbox_list[2][2][1]
    if min_y1b > max_y1b:
        min_y1b, max_y1b = max_y1b, min_y1b

    if rotate2 > 0:
        xlist = []
        for point in bbox_list[3]:
            xlist.append(point[0])
        index = xlist.index(min(xlist))
        min_x2 = bbox_list[3][index][0]
        xlist = []
        for point in bbox_list[1]:
            xlist.append(point[0])
        index = xlist.index(max(xlist))
        max_x2 = bbox_list[1][0][0]
        if min_x2 > max_x2:
            min_x2, max_x2 = max_x2, min_x2
        min_y2 = bbox_list[3][2][1] - rotate2 * 1.51
        max_y2 = bbox_list[1][0][1] + rotate2 * 1.51
        if min_y2 > max_y2:
            min_y2, max_y2 = max_y2, min_y2
    else:
        min_x2 = bbox_list[3][2][0] - rotate2 * 2.81
        max_x2 = bbox_list[1][0][0] + rotate2 * 2.81
        if min_x2 > max_x2:
            min_x2, max_x2 = max_x2, min_x2
        ylist = []
        for point in bbox_list[3]:
            ylist.append(point[0])
        index = ylist.index(min(ylist))
        min_y2 = bbox_list[3][index][1]
        ylist = []
        for point in bbox_list[0]:
            ylist.append(point[0])
        index = ylist.index(max(ylist))
        max_y2 = bbox_list[1][index][1]
        if min_y2 > max_y2:
            min_y2, max_y2 = max_y2, min_y2

    if min_x2 > max_x2 and rotate2 > -45:
        min_x1a, max_x1a, min_x1b, max_x1b = min_x1b, max_x1b, min_x1a, max_x1a
        
    if min_y2 > max_y2 and rotate2 < 45:
        min_y1a, max_y1a, min_y1b, max_y1b = min_y1b, max_y1b, min_y1a, max_y1a

    # Check for overlaps
    if max_x1a + 15 < min_x2 or max_x2 + 15 < min_x1a or max_y1a + 15 < min_y2 or max_y2 + 15 < min_y1a:
        if max_x1a > 720 or min_x1a < 0 or max_y1a > 720 or min_y1a < 0 or max_x1a < 0 or min_x1a > 720 or max_y1a < 0 or min_y1a > 720:
            a = False
        else:
            a = True
    else:
        a = False

    if max_x1b + 15 < min_x2 or max_x2 + 15 < min_x1b or max_y1b + 15 < min_y2 or max_y2 + 15 < min_y1b:
        if max_x1b > 720 or min_x1b < 0 or max_y1b > 720 or min_y1b < 0 or max_x1b < 0 or min_x1b > 720 or max_y1b < 0 or min_y1b > 720:
            b = False
        else:
            b = True
    else:
        b = False

    if bbox_list[1][0][1] > 720 or bbox_list[1][0][0] > 720 or bbox_list[1][0][1] < 0 or bbox_list[1][0][0] < 0 or bbox_list[1][2][1] > 720 or bbox_list[1][2][0] > 720 or bbox_list[1][2][1] < 0 or bbox_list[1][2][0] < 0:
        c = False
    else:
        c = True

    if bbox_list[3][2][1] < 0 or bbox_list[3][2][0] < 0 or bbox_list[3][2][1] > 720 or bbox_list[3][2][0] > 720 or bbox_list[3][0][1] < 0 or bbox_list[3][0][0] < 0 or bbox_list[3][0][1] > 720 or bbox_list[3][0][0] > 720:
        d = False
    else:
        d = True

    # Determine the valid bounding boxes and names
    if b and a:
        valid_names = [names[0], names[1]]
        if not c:
            valid_bboxes = [[bbox_list[0], bbox_list[2]], [bbox_list[1]]]
            l = 5
        elif not d:
            valid_bboxes = [[bbox_list[0], bbox_list[2]], [bbox_list[3]]]
            l = 6
        else:
            valid_bboxes = [[bbox_list[0], bbox_list[2]], [bbox_list[1], bbox_list[3]]]
            l = 1
    elif b and not a:
        valid_names = [names[0], names[1]]
        if not c:
            valid_bboxes = [[bbox_list[2]], [bbox_list[1]]]
            l = 7
        elif not d:
            valid_bboxes = [[bbox_list[2]], [bbox_list[3]]]
            l = 8
        else:
            valid_bboxes = [[bbox_list[2]], [bbox_list[1], bbox_list[3]]]
            l = 2
    elif not b and a:
        valid_names = [names[0], names[1]]
        if not c:
            valid_bboxes = [[bbox_list[0]], [bbox_list[1]]]
            l = 9
        elif not d:
            valid_bboxes = [[bbox_list[0]], [bbox_list[3]]]
            l = 10
        else:
            valid_bboxes = [[bbox_list[0]], [bbox_list[1], bbox_list[3]]]
            l = 3
    else:
        valid_names = [names[1]]
        if not c:
            valid_bboxes = [[bbox_list[1]]]
            l = 11
        elif not d:
            valid_bboxes = [[bbox_list[3]]]
            l = 12
        else:
            valid_bboxes = [[bbox_list[1], bbox_list[3]]]
            l = 4

    return valid_bboxes, valid_names, l