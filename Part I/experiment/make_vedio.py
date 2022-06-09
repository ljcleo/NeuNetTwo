from PIL import Image
import numpy as np
import cv2
import os

vedio_name = 'Seq05VD'
img_dir = '/root/final-term/InverseForm/result/'
size = (960,720)
image_list  = []
for file_name in os.listdir(img_dir):
    if 'colored' in file_name and vedio_name in file_name:
        image_list.append(file_name)
        # I = Image.open(img_dir+file_name) 
        # I_array = np.array(I)
        # print(I_array.shape)
image_list.sort()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('/root/final-term/InverseForm/result/mp4/{}.mp4'.format(vedio_name), fourcc, 5, size)

for i in range(len(image_list)):
    img = cv2.imread(img_dir+image_list[i]) 
    videoWriter.write(img)
videoWriter.release()



    

