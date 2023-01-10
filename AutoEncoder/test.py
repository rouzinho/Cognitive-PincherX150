#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class ConvolutionalAutoencoder(nn.Module):
  def __init__(self):
    super(ConvolutionalAutoencoder, self).__init__()

    # Encoder
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)

    # Latent space
    self.fc1 = nn.Linear(8 * 8 * 8, 8)

    # Decoder
    self.fc2 = nn.Linear(8, 8 * 8 * 8)
    self.deconv1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

  def forward(self, x):
    # Encoder
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    # Latent space
    x = x.view(-1, 8 * 8 * 8)
    x = self.fc1(x)

    # Decoder
    x = self.fc2(x)
    x = x.view(-1, 8, 8, 8)
    x = self.deconv1(x)
    x = self.deconv2(x)
    x = self.deconv3(x)

    return x

class CompressState(object):
    def __init__(self):
        self.ae = ConvolutionalAutoencoder()

    def trainModel(self):
        current_cost = 0
        current_test = 0
        last_cost = 15
        learning_rate = 5e-2
        epochs = 1000
        data_input = []

        self.ae.to(device)
        criterion = torch.nn.BCELoss()
        #optimizer = torch.optim.Adam(self.ae.parameters(),lr=learning_rate)
        optimizer = torch.optim.Adadelta(self.ae.parameters(),lr=learning_rate)
        #get inputs and targets

        img = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_1.jpg')
        t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = np.float32(t)
        res = res*(1/255.0)
        res = res[np.newaxis,...]
        print(res.shape)
        sample = torch.from_numpy(res)
        target = sample
        target = target.long()
        sample = sample.unsqueeze(0)
        sample = sample.to(device)
        for i in range(0,epochs):
            self.ae.train()
            current_cost = 0
            optimizer.zero_grad()
            dec = self.ae(sample)
            cost = criterion(dec,sample)
            cost.backward()
            optimizer.step()
            current_cost = current_cost + cost.item()
            print("Epoch: {}/{}...".format(i, epochs),
                                "Cross Entropy Training : ",current_cost)

    def getRepresentation(self,input):
      self.ae.eval()
      dec = self.ae(input)
      return dec


if __name__ == "__main__":
    torch.manual_seed(58)
    cs = CompressState()
    cs.trainModel()
    #cs.loadModel()
    img = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_1.jpg')
    t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = np.float32(t)
    res = res*(1/255.0)
    res2 = res[np.newaxis,...]
    
    sample = torch.from_numpy(res2)
    target = sample
    target = target.long()
    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    dec = cs.getRepresentation(sample)
    dec = torch.squeeze(dec)
    
    test = dec.cpu().detach().numpy()
    #plt.imshow(res)
    plt.imshow(test)
    plt.show()