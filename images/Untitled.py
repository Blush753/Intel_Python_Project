#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import collections
import numpy as np
import matplotlib.pyplot as plt
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


# In[3]:


im = cv2.imread("images\download.jpg")
cv2.imshow('my image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[156]:


class Predict:
    model=0
    def __init__(self,model):
        self.t=list()
        self.s=list()
        self.ts=list()
        self.n1=0
        self.n2=0
        self.n3=0
        self.n4=0
        self.cfg = get_cfg()
        self.model=model
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        if model==1:
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
           
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        elif model==2:
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
         
            
        elif model==3:
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

            
        elif model==4:
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
        self.cfg.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.cfg)
    
    def image(self,path):
        im = cv2.imread(path)
        if self.model != 4:
            
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            a=outputs["instances"].pred_classes.tolist()
            b=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
            self.ts = list(map(lambda x: b[x] , a))
            
            if self.model==1:
                self.n1=len(self.ts)
            if self.model==2:
                self.n2=len(self.ts)
            else:
                self.n3=len(self.ts)
            
            
           
            
        
        else:
            outputs, info = self.predictor(im)["panoptic_seg"]  #getting predictions, here info is a list that contains a dictionary for every predicted asset
            v = Visualizer(im[:, :, ::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2) #visualizing predictions
            out = v.draw_panoptic_seg_predictions(outputs.to("cpu"), info)  #drawing predictions over original image
            
            thing=[]  #empty list for things
            stuff=[]  #empty list for stuff doesn't count towards objects
            
            for i in range(0,len(info)): #loop to find things in info
                if info[i]['isthing']:
                    thing.append(info[i]['category_id'])
                    
            b=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes #metadata for things
            t = list(map(lambda x: b[x] , thing))   #reading the indexes from metadata to convert the data to string labels
            
            self.n4= len(t)  #counting the no. of objects
            print(t)
            
            for i in range(0,len(info)):  #loop to find stuff in info
                if info[i]['isthing']==False:
                    stuff.append(info[i]['category_id'])
                    
            b=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).stuff_classes  #metadata for stuff(such as wall,food,fruit,etc)
            s = list(map(lambda x: b[x] , stuff))  #reading the indexes from metadata to convert the data to string labels
            print(s)    
            
            self.ts=t+s  #joining the two lists for things and stuff
            
        
        cv2.imshow('image',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def display(self,list_name):
        print(self.n1)
        counter = collections.Counter(list_name)

        for key, value in counter.items():
            print("  "+f"{key} : {value}")
        


# In[157]:


p="images/3.jpg"
a=Predict(1)
a.image(p)
a.display(a.ts)




# In[ ]:


while True:
    # code to be executed goes here
    print("Hello, world!")
    
    # ask user if they want to run again
    run_again = input("Do you want to run again? (y/n) ").lower()
    
    # check user's response
    if run_again == 'n':
        break


# In[127]:


import collections
v=1
my_list = ['a', 'b', 'c', 'd', 'a', 'a', 'b']
counter = collections.Counter(my_list)

for key, value in counter.items():
    print("  "+f"{key} : {value}")
n.v=2
print(n.v)


# In[ ]:




