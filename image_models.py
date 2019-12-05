import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage.io import imread
from torchsummary import summary
import pandas as pd
import time


PAD = 1
PAD_MXP = 0
PAD_TRANSP = 0
BIAS = True
PRINT_SIZE = False 
MXP_MOM = 0.99
MXP_EPS = 0.001


class VAE_CNN(nn.Module):
    def __init__(self, BOTTLENECK_SIZE):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=(1,1), bias=BIAS)
        self.bn1 = nn.BatchNorm2d(32, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp1 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn2 = nn.BatchNorm2d(64, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp2 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn3 = nn.BatchNorm2d(128, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp3 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn4 = nn.BatchNorm2d(256, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp4 = nn.MaxPool2d(2, padding=1)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(13*13*256, BOTTLENECK_SIZE)
        self.fc_bn1 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        self.fc2 = nn.Linear(13*13*256, BOTTLENECK_SIZE)
        self.fc_bn2 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        # Sampling vector
        self.fc3 = nn.Linear(BOTTLENECK_SIZE, 13*13*256)
        #self.fc_bn3 = nn.BatchNorm1d(13*13*256)
        
        # Decoder
        self.ups5 = nn.Upsample(scale_factor = 2) 
        self.conv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn5 = nn.BatchNorm2d(256, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups6 = nn.Upsample(scale_factor = 2) 
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn6 = nn.BatchNorm2d(128, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups7 = nn.Upsample(scale_factor = 2) 
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn7 = nn.BatchNorm2d(64, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups8 = nn.Upsample(scale_factor = 2) 
        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn8 = nn.BatchNorm2d(32, eps = MXP_EPS, momentum = MXP_MOM )
        
        
        self.conv9 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=BIAS)
        

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        conv1 = self.mxp1(self.bn1(self.relu(self.conv1(x))))
        conv2 = self.mxp2(self.bn2(self.relu(self.conv2(conv1))))
        conv3 = self.mxp3(self.bn3(self.relu(self.conv3(conv2))))
        conv4 = self.mxp4(self.bn4(self.relu(self.conv4(conv3))))
        conv4 = conv4.view(-1, 13 * 13 * 256)

        fc1 = self.fc_bn1(self.fc1(conv4))
        fc2 = self.fc_bn2(self.fc2(conv4))
        
        '''
        print(conv1.size())
        print(conv2.size())
        print(conv3.size())
        print(conv4.size())
        print(conv4.size())
        print(fc1.size())
        print(fc2.size())
        '''
        return fc1, fc2

    def reparameterize(self, mu, logvar):
        '''        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        '''
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

        
    def decode(self, z):
        fc3 = self.fc3(z)
        fc3 = fc3.view(-1, 256, 13,13)
        
        conv5 = self.bn5(self.relu(self.conv5(self.ups5(fc3))))
        conv6 = self.bn6(self.relu(self.conv6(self.ups6(conv5))))
        conv6 = conv6.narrow(3, 0, 51)
        conv6 = conv6.narrow(2, 0, 51)
        conv7 = self.bn7(self.relu(self.conv7(self.ups7(conv6))))
        conv7 = conv7.narrow(3, 0, 100)
        conv7 = conv7.narrow(2, 0, 100)    
        conv8 = self.bn8(self.relu(self.conv8(self.ups8(conv7))))
        conv9 = self.tanh(self.conv9(conv8))

        '''
        print(fc3.size())
        print(fc3.size())
        print(conv5.size())
        print(conv6.size())
        print(conv7.size())
        print(conv8.size())
        print(conv9.size())'''
        
        return conv9

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

    

class VAE_CNN_antigo(nn.Module):
    def __init__(self, BOTTLENECK_SIZE):
        super(VAE_CNN_antigo, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=(1,1), bias=BIAS)
        self.bn1 = nn.BatchNorm2d(32)
        self.mxp1 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn2 = nn.BatchNorm2d(64)
        self.mxp2 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn3 = nn.BatchNorm2d(128)
        self.mxp3 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn4 = nn.BatchNorm2d(256)
        self.mxp4 = nn.MaxPool2d(2, padding=1)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(13*13*256, BOTTLENECK_SIZE)
        self.fc_bn1 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        self.fc2 = nn.Linear(BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        self.fc_bn2 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        # Sampling vector
        self.fc3 = nn.Linear(BOTTLENECK_SIZE, 13*13*256)
        self.fc_bn3 = nn.BatchNorm1d(13*13*256)
        
        # Decoder
        self.ups5 = nn.Upsample(scale_factor = 2) 
        self.conv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.ups6 = nn.Upsample(scale_factor = 2) 
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.ups7 = nn.Upsample(scale_factor = 2) 
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.ups8 = nn.Upsample(scale_factor = 2) 
        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn8 = nn.BatchNorm2d(32)
        
        
        self.conv9 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=BIAS)
        

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        conv1 = self.mxp1(self.relu(self.bn1(self.conv1(x))))
        conv2 = self.mxp2(self.relu(self.bn2(self.conv2(conv1))))
        conv3 = self.mxp3(self.relu(self.bn3(self.conv3(conv2))))
        conv4 = self.mxp4(self.relu(self.bn4(self.conv4(conv3))))
        conv4 = conv4.view(-1, 13 * 13 * 256)

        fc1 = self.fc_bn1(self.fc_bn1(self.fc1(conv4)))
        fc2 = self.fc_bn2(self.fc_bn1(self.fc2(fc1))) 
        

        return fc1, fc2

    def reparameterize(self, mu, logvar):


        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std).add_(mu)

        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc3 = fc3.view(-1, 256, 13,13)
        
        conv5 = self.relu(self.bn5(self.conv5(self.ups5(fc3))))
        #conv5 = conv5.narrow(3, 0, 26)
        #conv5 = conv5.narrow(2, 0, 26)
        conv6 = self.relu(self.bn6(self.conv6(self.ups6(conv5))))
        conv6 = conv6.narrow(3, 0, 51)
        conv6 = conv6.narrow(2, 0, 51)
        conv7 = self.relu(self.bn7(self.conv7(self.ups7(conv6))))
        conv7 = conv7.narrow(3, 0, 100)
        conv7 = conv7.narrow(2, 0, 100)    
        conv8 = self.relu(self.bn8(self.conv8(self.ups8(conv7))))
        conv9 = self.tanh(self.conv9(conv8))


        return conv9

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    
    

class VAE_CNN_no_bnorm(nn.Module):
    def __init__(self, BOTTLENECK_SIZE):
        super(VAE_CNN_no_bnorm, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=(1,1), bias=BIAS)
        self.bn1 = nn.BatchNorm2d(32, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp1 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn2 = nn.BatchNorm2d(64, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp2 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn3 = nn.BatchNorm2d(128, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp3 = nn.MaxPool2d(2, padding=PAD_MXP)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=PAD, bias=BIAS)
        self.bn4 = nn.BatchNorm2d(256, eps = MXP_EPS, momentum = MXP_MOM )
        self.mxp4 = nn.MaxPool2d(2, padding=1)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(13*13*256, BOTTLENECK_SIZE)
        self.fc_bn1 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        self.fc2 = nn.Linear(BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        self.fc_bn2 = nn.BatchNorm1d(BOTTLENECK_SIZE)
        
        # Sampling vector
        self.fc3 = nn.Linear(BOTTLENECK_SIZE, 13*13*256)
        self.fc_bn3 = nn.BatchNorm1d(13*13*256)
        
        # Decoder
        self.ups5 = nn.Upsample(scale_factor = 2) 
        self.conv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn5 = nn.BatchNorm2d(256, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups6 = nn.Upsample(scale_factor = 2) 
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn6 = nn.BatchNorm2d(128, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups7 = nn.Upsample(scale_factor = 2) 
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn7 = nn.BatchNorm2d(64, eps = MXP_EPS, momentum = MXP_MOM )
        
        self.ups8 = nn.Upsample(scale_factor = 2) 
        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=PAD, bias=BIAS)
        self.bn8 = nn.BatchNorm2d(32, eps = MXP_EPS, momentum = MXP_MOM )
        
        
        self.conv9 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=BIAS)
        

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        conv1 = self.mxp1(self.relu(self.conv1(x)))
        conv2 = self.mxp2(self.relu(self.conv2(conv1)))
        conv3 = self.mxp3(self.relu(self.conv3(conv2)))
        conv4 = self.mxp4(self.relu(self.conv4(conv3)))
        conv4 = conv4.view(-1, 13 * 13 * 256)

        fc1 = self.fc1(conv4)
        fc2 = self.fc2(fc1)
        
        '''
        print(conv1.size())
        print(conv2.size())
        print(conv3.size())
        print(conv4.size())
        print(conv4.size())
        print(fc1.size())
        print(fc2.size())
        '''
        return fc1, fc2

    def reparameterize(self, mu, logvar):
        '''        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        '''
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

        
    def decode(self, z):
        fc3 = self.fc_bn3(self.fc3(z))
        fc3 = fc3.view(-1, 256, 13,13)
        
        conv5 = self.relu(self.conv5(self.ups5(fc3)))
        conv6 = self.relu(self.conv6(self.ups6(conv5)))
        conv6 = conv6.narrow(3, 0, 51)
        conv6 = conv6.narrow(2, 0, 51)
        conv7 = self.relu(self.conv7(self.ups7(conv6)))
        conv7 = conv7.narrow(3, 0, 100)
        conv7 = conv7.narrow(2, 0, 100)    
        conv8 = self.relu(self.conv8(self.ups8(conv7)))
        conv9 = self.tanh(self.conv9(conv8))

        '''
        print(fc3.size())
        print(fc3.size())
        print(conv5.size())
        print(conv6.size())
        print(conv7.size())
        print(conv8.size())
        print(conv9.size())'''
        
        return conv9

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar