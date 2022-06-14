# NeuNetTwo --- Part II

Neural Network &amp; Deep Learning Final Project (Part II)

## Data Download 
[VOC Data](https://pan.baidu.com/s/1YuBbBKxm2FGgTU5OfaeC5A)  
Password: uack   

## Train Step
### a、Train with VOC07+12 data
1. Data Preparation   
**Before training, you need to download VOC07+12 dataset, unzip in the root directory**  

2. Preprocess   
Run voc_annotation.py and you will derive 2 files, 2007_train.txt and 2007_val.txt。   

3. Training  
You can modify train.py to train in different ways. For example,
- **Task a:** pretrain=False
- **Task b:** pretrain=True, put file _resnet50-19c8e357.pth_ into _./model_data/_ and change _./net/resnet50.py_ row 119 to
```
model.load_state_dict("./model_data/resnet50-19c8e357.pth")
```
- **Task c:** pretrain=True, and put file _coco_mask_r-cnn_res50.pth_ into _./model_data/_

4. Results
All results will be in logs. You can see all kind of loss and mAP.
