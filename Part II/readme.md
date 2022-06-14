# NeuNetTwo --- Part II

Neural Network &amp; Deep Learning Final Project (Part II)

## Requirement
torch == 1.2.0

## Data Download 
[VOC Data](https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A)  
Password: uack   

## Train Step
### a、Train with VOC07+12 data
1. Data Preparation   
**Before training, you need to download VOC07+12 dataset, unzip in the root directory**  

2. preprocess   
Run voc_annotation.py and you will derive 2 files, 2007_train.txt and 2007_val.txt。   

3. Train  
You can modify train.py to train in different ways. For example,
- **Task a:** pretrain=False
- **Task b:** pretrain=True, change _./net/resnet50.py_ row 119 to
```
state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
model.load_state_dict(state_dict)
```
- **Task c:** pretrain=True, and put file _coco_mask_r-cnn_res50.pth_ into _./model_data/_

4. Result
All results will be in logs. You can see all kind of loss and mAP.
