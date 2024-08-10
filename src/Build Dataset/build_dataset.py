from functions import get_random_background, get_random_cards, get_valid_bboxes
from generate_scene import Scene
import os
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches

name = '0'

print('Dataset generation Started')

for i in range(0, 1000):
   bg = get_random_background()
   img1, card_val1, hulla1, hullb1, index1 = get_random_cards()
   img2, card_val2, hulla2, hullb2, index2 = get_random_cards()

   test_scene = Scene(bg, img1, card_val1, hulla1, hullb1, img2, card_val2, hulla2, hullb2)

   valid_bboxes, valid_names, len = get_valid_bboxes(test_scene.bbox_list, test_scene.rotate1, 
                                                test_scene.rotate2, [card_val1, card_val2])

   #fig, ax = plt.subplots()
   #for bboxes in valid_bboxes:
      #for bbox in bboxes:
         #ax.plot([bbox[0][0], bbox[1][0]], [bbox[0][1], bbox[1][1]], linewidth=1, color='red')
         #ax.plot([bbox[1][0], bbox[2][0]], [bbox[1][1], bbox[2][1]], linewidth=1, color='red')
         #ax.plot([bbox[2][0], bbox[3][0]], [bbox[2][1], bbox[3][1]], linewidth=1, color='red')
         #ax.plot([bbox[3][0], bbox[0][0]], [bbox[3][1], bbox[0][1]], linewidth=1, color='red')
         #for point in bbox:
            #circle = patches.Circle((int(point[0]), int(point[1])), radius=0, edgecolor='green', facecolor='none', linewidth=2)
            #ax.add_patch(circle)
   #ax.imshow(test_scene.final)
   #ax.set_xlim([0, test_scene.final.shape[1]])
   #ax.set_ylim([test_scene.final.shape[0], 0])
   #plt.show(block=False)
   #plt.pause(4)
   #plt.close()
   
   name = str(int(name) + 1)
   label_dir_path = 'input/Card Detector/Dataset Versions/Dataset V4/Labels/test'

   os.makedirs(label_dir_path, exist_ok=True)
   file_path = os.path.join(label_dir_path, f'{name}.txt')
   print(name + ": " + str(len))

   with open(file_path, 'w') as f:
      if len == 1:
         f.write(f'{index1} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index1} {(valid_bboxes[0][1][0][0] + valid_bboxes[0][1][2][0]) / 1440} {(valid_bboxes[0][1][0][1] + valid_bboxes[0][1][1][1]) / 1440} {abs((valid_bboxes[0][1][2][0] - valid_bboxes[0][1][0][0])) / 720} {abs((valid_bboxes[0][1][1][1] - valid_bboxes[0][1][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][0][0][0] + valid_bboxes[1][0][2][0]) / 1440} {(valid_bboxes[1][0][0][1] + valid_bboxes[1][0][1][1]) / 1440} {abs((valid_bboxes[1][0][2][0] - valid_bboxes[1][0][0][0])) / 720} {abs((valid_bboxes[1][0][1][1] - valid_bboxes[1][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][1][0][0] + valid_bboxes[1][1][2][0]) / 1440} {(valid_bboxes[1][1][0][1] + valid_bboxes[1][1][1][1]) / 1440} {abs((valid_bboxes[1][1][2][0] - valid_bboxes[1][1][0][0])) / 720} {abs((valid_bboxes[1][1][1][1] - valid_bboxes[1][1][0][1])) / 720}\n')
      elif len == 2:
         f.write(f'{index1} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][0][0][0] + valid_bboxes[1][0][2][0]) / 1440} {(valid_bboxes[1][0][0][1] + valid_bboxes[1][0][1][1]) / 1440} {abs((valid_bboxes[1][0][2][0] - valid_bboxes[1][0][0][0])) / 720} {abs((valid_bboxes[1][0][1][1] - valid_bboxes[1][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][1][0][0] + valid_bboxes[1][1][2][0]) / 1440} {(valid_bboxes[1][1][0][1] + valid_bboxes[1][1][1][1]) / 1440} {abs((valid_bboxes[1][1][2][0] - valid_bboxes[1][1][0][0])) / 720} {abs((valid_bboxes[1][1][1][1] - valid_bboxes[1][1][0][1])) / 720}\n')
      elif len == 3:
         f.write(f'{index1} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][0][0][0] + valid_bboxes[1][0][2][0]) / 1440} {(valid_bboxes[1][0][0][1] + valid_bboxes[1][0][1][1]) / 1440} {abs((valid_bboxes[1][0][2][0] - valid_bboxes[1][0][0][0])) / 720} {abs((valid_bboxes[1][0][1][1] - valid_bboxes[1][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][1][0][0] + valid_bboxes[1][1][2][0]) / 1440} {(valid_bboxes[1][1][0][1] + valid_bboxes[1][1][1][1]) / 1440} {abs((valid_bboxes[1][1][2][0] - valid_bboxes[1][1][0][0])) / 720} {abs((valid_bboxes[1][1][1][1] - valid_bboxes[1][1][0][1])) / 720}\n')
      elif len == 4:
         f.write(f'{index2} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[0][1][0][0] + valid_bboxes[0][1][2][0]) / 1440} {(valid_bboxes[0][1][0][1] + valid_bboxes[0][1][1][1]) / 1440} {abs((valid_bboxes[0][1][2][0] - valid_bboxes[0][1][0][0])) / 720} {abs((valid_bboxes[0][1][1][1] - valid_bboxes[0][1][0][1])) / 720}\n')
      elif len == 5 or len == 6:
         f.write(f'{index1} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index1} {(valid_bboxes[0][1][0][0] + valid_bboxes[0][1][2][0]) / 1440} {(valid_bboxes[0][1][0][1] + valid_bboxes[0][1][1][1]) / 1440} {abs((valid_bboxes[0][1][2][0] - valid_bboxes[0][1][0][0])) / 720} {abs((valid_bboxes[0][1][1][1] - valid_bboxes[0][1][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][0][0][0] + valid_bboxes[1][0][2][0]) / 1440} {(valid_bboxes[1][0][0][1] + valid_bboxes[1][0][1][1]) / 1440} {abs((valid_bboxes[1][0][2][0] - valid_bboxes[1][0][0][0])) / 720} {abs((valid_bboxes[1][0][1][1] - valid_bboxes[1][0][0][1])) / 720}\n')
      elif len == 7 or len == 8 or len == 9 or len == 10:
         f.write(f'{index1} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')
         f.write(f'{index2} {(valid_bboxes[1][0][0][0] + valid_bboxes[1][0][2][0]) / 1440} {(valid_bboxes[1][0][0][1] + valid_bboxes[1][0][1][1]) / 1440} {abs((valid_bboxes[1][0][2][0] - valid_bboxes[1][0][0][0])) / 720} {abs((valid_bboxes[1][0][1][1] - valid_bboxes[1][0][0][1])) / 720}\n')
      elif len == 11 or len == 12:
         f.write(f'{index2} {(valid_bboxes[0][0][0][0] + valid_bboxes[0][0][2][0]) / 1440} {(valid_bboxes[0][0][0][1] + valid_bboxes[0][0][1][1]) / 1440} {abs((valid_bboxes[0][0][2][0] - valid_bboxes[0][0][0][0])) / 720} {abs((valid_bboxes[0][0][1][1] - valid_bboxes[0][0][0][1])) / 720}\n')

   image_dir_path = 'input/Card Detector/Dataset Versions/Dataset V4/Images/test'
   os.makedirs(image_dir_path, exist_ok=True)

   img_file_path = os.path.join(image_dir_path, f'{name}.jpg')
   cv2.imwrite(img_file_path, test_scene.final)

print('Dataset Generation Finished')