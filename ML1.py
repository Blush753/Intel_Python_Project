import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
from tkinter import Canvas
import time
import io
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

a,n1,n2,n3,n4,t1,t2,t3,t4=0,0,0,0,0,0,0,0,0
th1,th2,th3,th4,s,ts=[],[],[],[],[],[]
model=0
out = None
def predict(model):
    global out,path,n1,n2,n3,n4,t1,t2,t3,t4,th1,th2,th3,th4,s
    cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    if model==1:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    elif model==2:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
         
            
    elif model==3:
        cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

            
    elif model==4:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    ########################################################################################################################
    if path=="":
        path = filedialog.askopenfilename()
    im = cv2.imread(path)
    if model != 4:
        start_time = time.time()
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        end_time = time.time()
        o=outputs["instances"].pred_classes.tolist()
        b=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        
            
        if model==1:
            th1 = list(map(lambda x: b[x] , o))
            n1=len(th1)
            t1=end_time-start_time
            
        if model==2:
            th2 = list(map(lambda x: b[x] , o))
            n2=len(th2)
            t2=end_time-start_time
        else:
            th3 = list(map(lambda x: b[x] , o))
            n3=len(th3) 
            t3=end_time-start_time
        
    else:
        start_time = time.time()
        outputs, info = predictor(im)["panoptic_seg"]  #getting predictions, here info is a list that contains a dictionary for every predicted asset
        v = Visualizer(im[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) #visualizing predictions
        out = v.draw_panoptic_seg_predictions(outputs.to("cpu"), info)  #drawing predictions over original image
        end_time = time.time()
        t4=end_time-start_time
            
        thing=[]  #empty list for things
        stuff=[]  #empty list for stuff doesn't count towards objects
            
        for i in range(0,len(info)): #loop to find things in info
            if info[i]['isthing']:
                thing.append(info[i]['category_id'])
                    
        b=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes #metadata for things
        th4 = list(map(lambda x: b[x] , thing))   #reading the indexes from metadata to convert the data to string labels
            
        n4= len(th4)  #counting the no. of objects
        
        for i in range(0,len(info)):  #loop to find stuff in info
            if info[i]['isthing']==False:
                stuff.append(info[i]['category_id'])
                    
        b=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes  #metadata for stuff(such as wall,food,fruit,etc)
        s = list(map(lambda x: b[x] , stuff))  #reading the indexes from metadata to convert the data to string labels  


#setup
app = ctk.CTk()
app.title("Detectron")
app.geometry("1000x600")
app.minsize(800,500)
ctk.set_appearance_mode('dark')
zoom_value=1
#layout
app.rowconfigure(0,weight = 1)
app.columnconfigure(0,weight= 2,uniform='a')
app.columnconfigure(1,weight= 6,uniform='a')
#widgets
photo=None
def file_path():
    global path,image,image1,button1,image2,image3,image4,Original
    button1.grid_forget()
    path=""
    path = filedialog.askopenfilename()
    
    for i in range(1,5):
        button1.grid_forget()
        predict(i)
        a=out.get_image()
        image = Image.fromarray(a)       # create an Image object from the image array
        if i ==1:
            image1=image
        if i ==2:
            image2=image
        if i ==3:
            image3=image
        if i ==4:
            image4 = image
    get_graph(n1,n2,n3,n4,t1,t2,t3,t4) 
    #gtoImage()
    #plt.close(fig)
    Original = Image.open(path)
    image=Original
    button1.grid_forget()            
    frame1.grid_forget()
    reset()    
def select_image():
    global frame1,button1
    frame1=ctk.CTkFrame(app,fg_color='#242424')
    frame1.grid(column=0,columnspan=2,row=0,sticky='nsew')
    button1=ctk.CTkButton(app, text="Select Image", command=file_path)
    button1.grid(row=0, column=0, columnspan=2)
def restart(): 
    canvas.delete('all')
    button3.place_forget()
    select_image()
def canvas_image():
    global photo,image,canvas,image_ratio,image,event_width,event_height,image,image_width,image_height,image5
    image_ratio = image.size[0]/image.size[1]
    
    canvas=Canvas(app, background='#242424', bd=0,highlightthickness=0, relief='ridge')
    canvas.grid(row=0,column=1,sticky='nsew', padx=3,pady=3)
    canvas.bind('<Configure>', show_image)
    canvas_ratio = event_width/event_height
    if canvas_ratio > image_ratio: #canvas is wider
        image_height = int(event_height)
        image_width = int(image_height*image_ratio)
    else:
        image_width = int(event_width)
        image_height = int(image_width/image_ratio)

    resized_image = image.resize((image_width,image_height))
    photo = ImageTk.PhotoImage(resized_image)
def show_image(event):
    global event_width,event_height,button3,photo
    event_width = event.width
    event_height =event.height
    canvas.delete('all') # deleting instances when resizing the window
    canvas.create_image(event.width/2,event.height/2, image=photo)  #centering the image
    button3 = ctk.CTkButton(app, text='X', text_color ='#FFF', fg_color = 'transparent',width = 40, height=40, hover_color='#8a0606',  command=restart)
    button3.place(relx=0.99, rely=0.01, anchor = 'ne')

#button = ctk.CTkButton(app, text="my button", command=canvas_image)
#button.grid(row=0, column=0, padx=20, pady=20)
##############  adding the tabs and their frames   ##############################
menu=ctk.CTkTabview(app)
menu.grid(row=0,column=0,sticky='nsew',pady=8,padx=8)
menu.add('Image')
menu.add('Info')
image_frame=ctk.CTkFrame(menu.tab('Image'),fg_color='transparent')
image_frame.pack(expand = True,fill='both')
info_frame=ctk.CTkFrame(menu.tab('Info'),fg_color='transparent')
info_frame.pack(expand = True,fill='both') 
################## image TAB   #########################
################  zoom slider #####################################################
zoom=ctk.CTkFrame(image_frame,fg_color = '#4a4a4a')
zoom.pack(fill = 'x',pady=4,ipady=8)
zoom.rowconfigure((0,1),weight = 1)
zoom.columnconfigure((0,1),weight = 1)
ctk.CTkLabel(zoom,text='Zoom').grid(column=0,row=0,sticky='w',padx=8)
zoom_label=ctk.CTkLabel(zoom,text="1.0")
zoom_label.grid(column=1,row=0,sticky='e',padx=8)

def zoom_(value):
    global zoom_value,image,image_width,image_height,canvas,photo
    zoom_value=round(value,1)
    zoom_label.configure(text = zoom_value)
    if zoom_value>0:
        image_width1=image_width*zoom_value
        image_height1=image_height*zoom_value
        image_width2=int(image_width1)
        image_height2=int(image_height1)
    resized_image = image.resize((image_width2,image_height2))
    photo = ImageTk.PhotoImage(resized_image)
    canvas.delete('all') # deleting instances when resizing the window
    canvas.create_image(event_width/2,event_height/2, image=photo)
ctk.CTkSlider(zoom, fg_color='#64686b',from_ = 0.1,to=2,command=zoom_).grid(row=1,column=0,columnspan=2)

show_=ctk.CTkFrame(image_frame,fg_color = '#4a4a4a')
show_.pack(fill = 'x',pady=4,ipady=8)
show_.rowconfigure((0,1,2,3,4),weight = 1)
show_.columnconfigure((0),weight = 1)
ctk.CTkLabel(show_,text='Show',font=('Bodoni',14)).grid(column=0,row=0)

def combobox_callback(choice):
    global Original,image1,image2,image3,image4,image,th1,t1,n1,s,fig
    print("combobox dropdown clicked:", choice)
    if choice=="Original":
        textbox.delete("1.0", "end")
        image = Original
        canvas_image()
        I.set("")
        L.set("")
        P.set("")
    elif choice=="Object Detection":
        textbox.delete("1.0", "end")
        image = image1
        canvas_image()
        I.set("")
        L.set("")
        P.set("")
        text_ = ""
        counter = collections.Counter(th1)
        for key, value in counter.items():
            text_ += "~~> "+f"{key} : {value}"+"\n"
        textbox.insert("1.0", "Total time taken = "+(str(round(t1,6)))+" sec\n")
        textbox.insert("2.0", "Total No. of objects = "+(str(n1))+"\n\n")
        textbox.insert("4.0", "List of Objects :~"+"\n")
        textbox.insert("5.0", text_)
    elif choice=="Instance Segmentation":
        textbox.delete("1.0", "end")
        image = image2
        canvas_image()
        L.set("")
        P.set("")
        O.set("")
        text_ = ""
        counter = collections.Counter(th2)
        for key, value in counter.items():
            text_ += "~~> "+f"{key} : {value}"+"\n"
        textbox.insert("1.0", "Total time taken = "+(str(round(t2,6)))+" sec\n")
        textbox.insert("2.0", "Total No. of objects = "+(str(n2))+"\n\n")
        textbox.insert("4.0", "List of Objects :~"+"\n")
        textbox.insert("5.0", text_)
    elif choice=="LVIS Instance Segmentation":
        textbox.delete("1.0", "end")
        image = image3
        canvas_image()
        I.set("")
        P.set("")
        O.set("")
        text_ = ""
        counter = collections.Counter(th3)
        for key, value in counter.items():
            text_ += "~~> "+f"{key} : {value}"+"\n"
        textbox.insert("1.0", "Total time taken = "+(str(round(t3,6)))+" sec\n")
        textbox.insert("2.0", "Total No. of objects = "+(str(n3))+"\n\n")
        textbox.insert("4.0", "List of Objects :~"+"\n")
        textbox.insert("5.0", text_)
    elif choice=="Panoptic Segmentation":
        textbox.delete("1.0", "end")
        image = image4
        canvas_image()
        I.set("")
        L.set("")
        O.set("")
        text_ = ""
        text_1 = ""
        counter = collections.Counter(th4)
        for key, value in counter.items():
            text_ += "~~> "+f"{key} : {value}"+"\n"
        counter = collections.Counter(s)
        for key, value in counter.items():
            text_1 += "~~> "+f"{key} : {value}"+"\n"
        textbox.insert("1.0", "Total time taken = "+(str(round(t4,6)))+" sec\n")
        textbox.insert("2.0", "Total No. of objects = "+(str(n4))+"\n\n")
        textbox.insert("4.0", "List of Objects :~"+"\n")
        textbox.insert("5.0", text_+"\n")
        textbox.insert(ctk.END, "List of Stuff :~"+"\n")
        textbox.insert(ctk.END, text_1)
        
O=ctk.CTkSegmentedButton(show_, values=["Original", "Object Detection"],dynamic_resizing=True,command=combobox_callback)
O.grid(row=1)
I=ctk.CTkSegmentedButton(show_, values=["Instance Segmentation"],dynamic_resizing=True,command=combobox_callback)
I.grid(row=2)
L=ctk.CTkSegmentedButton(show_, values=["LVIS Instance Segmentation"],dynamic_resizing=True,command=combobox_callback)
L.grid(row=3)
P=ctk.CTkSegmentedButton(show_, values=["Panoptic Segmentation"],dynamic_resizing=True,command=combobox_callback)
P.grid(row=4)


def reset():
    global Original,image
    image = Original
    canvas_image()
    O.set("Original")
    I.set("")
    L.set("")
    P.set("")
    menu.set('Image')

button = ctk.CTkButton(image_frame, text="Reset image",fg_color = '#4a4a4a', command=reset)
button.pack(fill = 'x',pady=4,ipady=8)
textbox = ctk.CTkTextbox(image_frame,fg_color='grey', text_color='black',height=600)
textbox.pack(fill = 'both',pady=4,ipady=0)
################################## INFO TAB ################################################
def get_graph(n1,n2,n3,n4,t1,t2,t3,t4):
    global image5,image6
    def fig2img(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    def addtext(x,y):
        for i in range(len(x)):
            plt.text(x[i],y[i],round(y[i],1), ha = 'center',bbox = dict(facecolor = 'white', alpha =0.6))
    x = [n1, n2, n3, n4]
    y = [t1, t2, t3, t4]
    # create bar plot
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['OD', 'IS', 'LVIS', 'PS']
    plt.bar(x, y, color=colors, tick_label=x, label=labels, alpha =0.6)
    addtext(x,y)
    # set title and axis labels
    plt.title("Comparision of Current Image")
    plt.xlabel("Number of objects")
    plt.ylabel("Time taken (seconds)")
    # add legend
    plt.legend()
    fig1 = plt.gcf()
    image5 = fig2img(fig1)
    plt.close(fig1)
    #######-------------------------------------------------------------------------------------------------################
    data = {'n1': [n1],'n2': [n2],'n3': [n3],'n4': [n4],'t1': [t1],'t2': [t2],'t3': [t3],'t4': [t4]}
    # Make data frame of above data
    df1 = pd.DataFrame(data)
    # append data frame to CSV file
    df1.to_csv('info.csv', mode='a', index=False, header=False)
    # store the data in variables
    df = pd.read_csv('info.csv')
    plt.scatter(df['n1'], df['t1'], color='blue', label='OD')
    plt.scatter(df['n2'], df['t2'], color='red', label='IS')
    plt.scatter(df['n3'], df['t3'], color='green', label='LVIS')
    plt.scatter(df['n4'], df['t4'], color='orange', label='PS')
    plt.xlabel('Number of objects')
    plt.ylabel('Time taken (seconds)')
    plt.legend()
    fig2 = plt.gcf()
    image6 = fig2img(fig2)
    plt.close(fig2)
def something(choice):
    global image,image5,image6
    if choice =="Bar Graph":
        image=image5
        canvas_image()
    if choice =="Scatter plot":
        image=image6
        canvas_image()
    if choice =="Clear CSV":
        df = pd.read_csv('info.csv')
        # drop all rows from the dataframe
        df = df.iloc[0:0]
        df.to_csv('info.csv', index=False)
B=ctk.CTkSegmentedButton(info_frame, values=["Bar Graph","Scatter plot","Clear CSV"],dynamic_resizing=True,command=something)
B.grid(row=1)

select_image()

app.mainloop()
