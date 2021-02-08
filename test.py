import cv2
import os
import argparse

import torch
import torch.nn as nn
from PIL import Image

from network import CNN
import numpy as np
import matplotlib.pyplot as plt

using_opencv = False

if using_opencv:
    from opencv_transforms import transforms
else:
    from torchvision import transforms
    
# load pytorch model
c, h, w = 3, 64, 64
model = CNN(c, h, w)
model.load_state_dict( torch.load('model.pkl') )
class_name = ['bad', 'good', 'ugly']

# load transforms needed
transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

file = 'test_img.jpg'
# file = 'data/good/000000.jpg'
if using_opencv:
    oreo = cv2.imread(file)
    oreo = cv2.cvtColor(oreo, cv2.COLOR_BGR2RGB)
    # print(oreo[100,100])
else:
    oreo = Image.open(file)
    # pix = np.array(oreo.getdata()).reshape(oreo.size[0], oreo.size[1], 3)
    # print(pix[100,100])
oreo = transform( oreo ).unsqueeze_(0)

# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(np.transpose(oreo.squeeze(), (1,2,0))[:,:,i])
# plt.show()

result = model( oreo )
print(result)
result = result.argmax(1).item()
print(class_name[result])