# Visual Inspection
Classify Oreos as good, bad or ugly using computer vision and deep learning. Done by Devon Webb and Easton Potokar. 

Found on github: https://github.com/contagon/VisualInspection.

## Item Criteria
We define examples of good, bad and ugly as follows:
| good | ugly | bad |
|:---:|:---:|:---:|
| ![](examples/good.jpg) |![](examples/ugly.jpg)  |![](examples/bad.jpg) |

## Locating Objects
To locate the oreos, we simply took an image when beginning recording and set it as our "base". After that we did some slight blurring, differencing with the base, and thresholding to locate where changes or "oreos" in the image were. The code is in `run.py`.

## Classifying Objects
To classify the oreos, we used a Convolutional Neural Network. It took in a 64x64 image with 3 channels and output 3 probabibilities of being good, bad or ugly. Training parameters can be found in `train.py`, as well as a few transforms before inputting images into the network.

## Gathering Data
The most tedious/difficult part was gathering enough data to train the network. This was done by recording videos of different oreas (the ones seen above) and using a small string, spinning them slightly to get all possible angles of them. The code is in `gather_oreos.py`. The resulting data can be found on box: https://byu.box.com/s/xzd5lf48uvfndyrhkowjfkglrmmyrldx

## Results
The results can be seen on youtube. Here's the link: https://youtu.be/clYLeb-u6Mo.