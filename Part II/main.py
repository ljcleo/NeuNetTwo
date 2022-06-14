import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch
import pickle
para=torch.load("./model_data/resnet50-19c8e357.pth")


file=open("./model_data/model_final_f10217.pkl","rb")
data=pickle.load(file)
file.close()

true_name=list(para.keys())
clean_name=list(data["model"].keys())[16:267+16]
change_index=[5,55,120,215]
put_index=[10,60,125,220]
for i in range(4):
    temp=clean_name[change_index[i]:change_index[i]+5]
    clean_name[change_index[i]:change_index[i]+15]=clean_name[put_index[i]:put_index[i]+15]
    clean_name[put_index[i]+10:put_index[i] + 15]=temp

for i in range(0,265,5):
    temp=clean_name[i+3:i+5]
    clean_name[i+3:i+5]=clean_name[i+1:i+3]
    clean_name[i + 1:i + 3]=temp

a=list(zip(true_name,clean_name))
for b in a:
    print(b)

for i in range(265):
    if para[true_name[i]].shape!=data["model"][clean_name[i]].shape:
        print(para[true_name[i]].shape,data["model"][clean_name[i]].shape)
    else:
        para[true_name[i]]=torch.tensor(data["model"][clean_name[i]])


torch.save(para,"./model_data/coco_mask_r-cnn_res50.pth")






