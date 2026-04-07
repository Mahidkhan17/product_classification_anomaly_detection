# Libraries
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as trans
import cnn_arch

# Image Transformer (Image Pixels --> Tensors)
img_transformer = trans.ToTensor()

# Model Path
data_folder_path = 'data/'
pc_model_path = data_folder_path + 'model/PC_MODEL_CEL.pt'
capsule_model_path = data_folder_path + 'model/CAPSULE_MODEL_CEL.pt'
leather_model_path = data_folder_path + 'model/LEATHER_MODEL_CEL.pt'
screw_model_path = data_folder_path + 'model/SCREW_MODEL_CEL.pt'

# Class-labels
products_labels = ['capsule', 'leather', 'screw']
condition_labels = ['defective', 'good']

# CNN Model Configs
pc_in_channels = 1
pc_out_channels = 3
dc_in_channels = 1
dc_out_channels = 2

# CNN Instances
PC_NET = cnn_arch.cnn_model(pc_in_channels, pc_out_channels).to('cpu')
CAPSULE_NET = cnn_arch.cnn_model_2(dc_in_channels, dc_out_channels).to('cpu')
LEATHER_NET = cnn_arch.cnn_model_2(dc_in_channels, dc_out_channels).to('cpu')
SCREW_NET = cnn_arch.cnn_model_2(dc_in_channels, dc_out_channels).to('cpu')

# Loading model's parameters (configurations, weights, & biases)
# Loading model's parameters (configurations, weights, & biases)

if(os.path.exists(pc_model_path)):
    PC_NET = torch.load(pc_model_path, map_location='cpu', weights_only=False)
    print('Product Classification Model is Loaded ({}).'.format(pc_model_path))
else:
    print('Unable to load model !!')
    exit(0)

if(os.path.exists(capsule_model_path)):
    CAPSULE_NET = torch.load(capsule_model_path, map_location='cpu', weights_only=False)
    print('Capsule Defect Model is Loaded ({}).'.format(capsule_model_path))
else:
    print('Unable to load model !!')
    exit(0)

if(os.path.exists(leather_model_path)):
    LEATHER_NET = torch.load(leather_model_path, map_location='cpu', weights_only=False)
    print('Leather Defect Model is Loaded ({}).'.format(leather_model_path))
else:
    print('Unable to load model !!')
    exit(0)

if(os.path.exists(screw_model_path)):
    SCREW_NET = torch.load(screw_model_path, map_location='cpu', weights_only=False)
    print('Screw Defect Model is Loaded ({}).'.format(screw_model_path))
else:
    print('Unable to load model !!')
    exit(0)
# Product class prediction function
def product_class_predict(img):
    # Resize Image to 100x100
    img = cv2.resize(img, (100, 100))
    # Colour Coversion : RGB to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image form to tensor
    x = img_transformer(img)
    x = x.unsqueeze(0)
    # Feedforward into classifier
    y = PC_NET.forward(x)
    # Convert tensor array to numpy
    y = y.detach().numpy()[0]
    # Get largest of all : arg-max
    y_amax = np.argmax(y)
    # Return predictions
    return(y, y_amax, products_labels[y_amax])

# Capsule defect prediction function
def capsule_defect_predict(img):
    # Resize Image to 100x100
    img = cv2.resize(img, (100, 100))
    # Colour Coversion : RGB to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image form to tensor
    x = img_transformer(img)
    x = x.unsqueeze(0)
    # Feedforward into classifier
    y = CAPSULE_NET.forward(x)
    # Convert tensor array to numpy
    y = y.detach().numpy()[0]
    # Get largest of all : arg-max
    y_amax = np.argmax(y)
    # Return predictions
    return(y, y_amax, condition_labels[y_amax])

# Leather defect prediction function
def leather_defect_predict(img):
    # Resize Image to 100x100
    img = cv2.resize(img, (100, 100))
    # Colour Coversion : RGB to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image form to tensor
    x = img_transformer(img)
    x = x.unsqueeze(0)
    # Feedforward into classifier
    y = LEATHER_NET.forward(x)
    # Convert tensor array to numpy
    y = y.detach().numpy()[0]
    # Get largest of all : arg-max
    y_amax = np.argmax(y)
    # Return predictions
    return(y, y_amax, condition_labels[y_amax])

# Screw defect prediction function
def screw_defect_predict(img):
    # Resize Image to 100x100
    img = cv2.resize(img, (100, 100))
    # Colour Coversion : RGB to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image form to tensor
    x = img_transformer(img)
    x = x.unsqueeze(0)
    # Feedforward into classifier
    y = SCREW_NET.forward(x)
    # Convert tensor array to numpy
    y = y.detach().numpy()[0]
    # Get largest of all : arg-max
    y_amax = np.argmax(y)
    # Return predictions
    return(y, y_amax, condition_labels[y_amax])
