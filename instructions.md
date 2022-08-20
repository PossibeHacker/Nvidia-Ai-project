# Nvidia-Ai-project

This project was made by me using the Jupyter notebook and their interface for training image classification models. I used a usb webcam for the camera in the code.

start off by logging into the jupyter notebook and then make a new folder.

download the dataset.py and utils.py files and paste them into the directory

make a new notebook and copy and paste the code from hand_gesture_recognition.py 





if you want to edit the amount of gestures look for this section of code

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

TASK = 'Gestures'
# TASK = 'emotions'
# TASK = 'fingers'
# TASK = 'diy'

CATEGORIES = ['Gestures_left', 'gestures_right', 'gestures_null','gestures_pause','gestures_unpause']
# CATEGORIES = ['none', 'happy', 'sad', 'angry']
# CATEGORIES = ['1', '2', '3', '4', '5']
# CATEGORIES = [ 'diy_1', 'diy_2', 'diy_3']

DATASETS = ['A', 'B']
# DATASETS = ['A', 'B', 'C']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset('../data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)
    
print("{} task with {} categories defined".format(TASK, CATEGORIES))
 
 
and edit the Categories tab but keep the gestures_null portion.

Now to train your ai run the tab and wait until the training widget shows up. first you want to set the category to the gestures_null category and take 50
pictures of the room your in without anything moving in the background. Next you can take the same amount of pictures for the other categories but with your 
hand in different shapes for each category. (as a tip you will get a better result if you take pictures of your hand in different spaces in the camera frame while
keeping it in the same shape.) 

Here is a video tutorial of my network working

https://youtu.be/SuSWUBafmIg

