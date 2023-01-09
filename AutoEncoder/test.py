import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
  def __init__(self):
    super(ConvolutionalAutoencoder, self).__init__()

    # Encoder
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
    self.batch_norm1 = nn.BatchNorm2d(16)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(8)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    self.batch_norm3 = nn.BatchNorm2d(8)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Latent space
    self.fc1 = nn.Linear(8 * 8 * 8, 8)

    # Decoder
    self.fc2 = nn.Linear(8, 8 * 8 * 8)
    self.deconv1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.batch_norm4 = nn.BatchNorm2d(8)
    self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.batch_norm5 = nn.BatchNorm2d(16)
    self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

  def forward(self, x):
    # Encoder
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.batch_norm3(x)
    x = self.pool3(x)

    # Latent space
    x = x.view(-1, 8 * 8 * 8)
    x = self.fc1(x)

    # Decoder
    x = self.fc2(x)
    x = x.view(-1, 8, 8, 8)
    x = self.deconv1(x)
    x = self.batch_norm
