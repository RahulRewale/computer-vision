## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.max_pool = nn.MaxPool2d(3, stride=2)
        
        # self.conv_drop1 = nn.Dropout(0.2) # using dropout after conv is not giving good output
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        # self.conv_drop2 = nn.Dropout(0.2) 
        
        self.conv3 = nn.Conv2d(64, 128, 5)
        # self.conv_drop3 = nn.Dropout(0.2)
        
        # self.conv4 = nn.Conv2d(128, 256, 5) # output is good even if the fourth conv layer is not present
        # self.conv_drop3 = nn.Dropout(0.2)
        
        # using fewer conv. layers leads to more in_features for the first dense layer
        # this results in more weights (i.e. parameters), and thus leading to 'out of memory' errors
        self.dense1 = nn.Linear(67712, 2048)
        self.drop1 = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(1024, 136)
        
        self.initialize_model()

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.max_pool(F.relu(self.conv1(x)))
        # x = self.conv_drop1(x)
        
        x = self.max_pool(F.relu(self.conv2(x)))
        # x = self.conv_drop2(x)
        
        x = self.max_pool(F.relu(self.conv3(x)))
        # x = self.conv_drop3(x)
        
        # x = self.max_pool(F.relu(self.conv4(x)))
        # x = self.conv_drop4(x)
        
        x = x.view(x.size(0), -1) 
        # print(x.size())  # to find out the in_features value for the first dense layer
        
        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        
        x = F.relu(self.dense2(x))
        x = self.drop2(x)
        
        x = self.dense3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    def initialize_model(self):
        # reference: https://www.youtube.com/watch?v=xWQ-p_o0Uik
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                I.xavier_uniform_(module.weight)
                I.constant_(module.bias, 0)