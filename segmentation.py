import torch
import torch.nn as nn
import cv2
import numpy as np
from .paths import *
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class ConvLRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.activation(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            ConvLRelu(in_channels, out_channels),
            ConvLRelu(out_channels, out_channels),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = DoubleConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        before_pool = self.conv_block(x)
        x = self.max_pool(before_pool)
        return x, before_pool


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_block = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, y):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv_block(torch.cat([x, y], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = EncoderBlock(in_channels, n_filters)
        self.enc2 = EncoderBlock(n_filters, n_filters * 2)
        self.enc3 = EncoderBlock(n_filters * 2, n_filters * 4)
        self.enc4 = EncoderBlock(n_filters * 4, n_filters * 8)

        self.center = DoubleConvBlock(n_filters * 8, n_filters * 16)

        self.dec4 = DecoderBlock(n_filters * (16 + 8), n_filters * 8)
        self.dec3 = DecoderBlock(n_filters * (8 + 4), n_filters * 4)
        self.dec2 = DecoderBlock(n_filters * (4 + 2), n_filters * 2)
        self.dec1 = DecoderBlock(n_filters * (2 + 1), n_filters)

        self.final = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.float()
        x, enc1 = self.enc1(x)
        x, enc2 = self.enc2(x)
        x, enc3 = self.enc3(x)
        x, enc4 = self.enc4(x)

        center = self.center(x)

        dec4 = self.dec4(center, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        final = self.final(dec1)

        return final

import cv2

def mask2boxes(PATH_TO_IMAGE, PATH_TO_MASK,return_crops=False):
  '''
  params
  ---
  PATH_TO_IMAGE : str
  PATH_TO_MASK : str

  returns
  ---
  boxes : list of tuples
    tuple is (y1,y2,x1,x2) where (y1,x1) is the first point of a box and (y2,x2) is the second one
  crops : list of numpy.ndarray
    correspondent segments from the image
  '''
  image = cv2.imread(PATH_TO_MASK)
  gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
  contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

  img = cv2.imread(PATH_TO_IMAGE)
  img = cv2.resize(img,dsize=(512,512))

  boxes = []
  crops = []

  for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    if abs(w) > 15 and abs(h) > 15: # skip if it is too small
      y1,y2=y-5,y+h+5
      x1,x2=x-5,x+w+5
      crop_img = img[y1:y2, x1:x2]
      boxes.append((y1,y2,x1,x2))
      crops.append(crop_img)
  if return_crops:
    return boxes, crops
  return boxes

def run_segmentation(model,PATH_TO_IMAGE):
    transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image = Image.open(PATH_TO_IMAGE).convert('RGB')
    image = transform(image)
    preds = torch.sigmoid(model(image.unsqueeze(0)))
    mask = (preds > 0.9999).float()
    save_image(mask, PATH_TO_MASK)
    boxes, crops = mask2boxes(PATH_TO_IMAGE,PATH_TO_MASK,return_crops=True)
    return boxes, crops


hyperparametrs = {
    'n_filters': 32,
}