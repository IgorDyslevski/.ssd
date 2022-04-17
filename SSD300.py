#!/usr/bin/env python
# coding: utf-8

# # Import

from torch import nn
from torchvision.models import mobilenet_v2
import torch
import numpy as np
import matplotlib.pyplot as plt


# # Model

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # uploading MobileNet
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet = self.mobilenet.features
    
    def forward(self, x):
        # first out -- N, 32, 38, 38
        first = self.mobilenet[:-12](x)
        # second out -- N, 96, 19, 19
        second = self.mobilenet[-12:-5](first)
        # third out -- N, 1280, 10, 10
        third = self.mobilenet[-5:](second)
        
        return first, second, third

    
class AuxiliaryBone(nn.Module):
    def __init__(self, convolution_filters):
        super().__init__()

        self.convolution_filters = convolution_filters
        
        # fourth out from BackBone
        self.fourth1 = nn.Conv2d(1280, self.convolution_filters[3], (3, 3), padding=1)
        self.fourth2 = nn.Conv2d(self.convolution_filters[3], self.convolution_filters[3], (3, 3), padding=1,
                                 stride=2)
        
        # fifth out
        self.fifth1 = nn.Conv2d(self.convolution_filters[3], self.convolution_filters[4], (3, 3), padding=1)
        self.fifth2 = nn.Conv2d(self.convolution_filters[4], self.convolution_filters[4], (3, 3), padding=1,
                                stride=2)
        
        # sixth out 
        self.sixth1 = nn.Conv2d(self.convolution_filters[4], self.convolution_filters[5], (3, 3), padding=1)
        self.sixth2 = nn.Conv2d(self.convolution_filters[5], self.convolution_filters[5], (3, 3),
                                stride=2)
        
    def forward(self, x):
        # fourth out -- N, 256, 5, 5
        fourth = self.fourth2(self.fourth1(x))
        # fifth out -- N, 256, 3, 3
        fifth = self.fifth2(self.fifth1(fourth))
        # sixth out -- N, 128, 1, 1
        sixth = self.sixth2(self.sixth1(fifth))
        return fourth, fifth, sixth


class PredictionBone(nn.Module):
    def __init__(self, aspect_ratios, sizes, convolution_filters, classes):
        super().__init__()

        self.aspect_ratios, self.sizes, self.convolution_filters, self.classes = aspect_ratios, sizes, \
                                                                                 convolution_filters, classes
        
        # location convolutions
        self.loc1 = nn.Conv2d(self.convolution_filters[0], len(self.aspect_ratios[0]) * 4, kernel_size=(3, 3),
                              padding=1)
        self.loc2 = nn.Conv2d(self.convolution_filters[1], len(self.aspect_ratios[1]) * 4, kernel_size=(3, 3),
                              padding=1)
        self.loc3 = nn.Conv2d(self.convolution_filters[2], len(self.aspect_ratios[2]) * 4, kernel_size=(3, 3),
                              padding=1)
        self.loc4 = nn.Conv2d(self.convolution_filters[3], len(self.aspect_ratios[3]) * 4, kernel_size=(3, 3),
                              padding=1)
        self.loc5 = nn.Conv2d(self.convolution_filters[4], len(self.aspect_ratios[4]) * 4, kernel_size=(3, 3),
                              padding=1)
        self.loc6 = nn.Conv2d(self.convolution_filters[5], len(self.aspect_ratios[5]) * 4, kernel_size=(3, 3),
                              padding=1)
        
        # class convolutions
        self.conf1 = nn.Conv2d(self.convolution_filters[0], len(self.aspect_ratios[0]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
        self.conf2 = nn.Conv2d(self.convolution_filters[1], len(self.aspect_ratios[1]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
        self.conf3 = nn.Conv2d(self.convolution_filters[2], len(self.aspect_ratios[2]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
        self.conf4 = nn.Conv2d(self.convolution_filters[3], len(self.aspect_ratios[3]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
        self.conf5 = nn.Conv2d(self.convolution_filters[4], len(self.aspect_ratios[4]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
        self.conf6 = nn.Conv2d(self.convolution_filters[5], len(self.aspect_ratios[5]) * len(self.classes),
                             kernel_size=(3, 3), padding=1)
    
    def forward(self, first, second, third, fourth, fifth, sixth):
        firstl = nn.Tanh()(self.loc1(first))
        firstl = firstl.permute(0, 2, 3, 1).contiguous().view(firstl.size(0), -1, 4)
        
        secondl = nn.Tanh()(self.loc2(second))
        secondl = secondl.permute(0, 2, 3, 1).contiguous().view(secondl.size(0), -1, 4)
        
        thirdl = nn.Tanh()(self.loc3(third))
        thirdl = thirdl.permute(0, 2, 3, 1).contiguous().view(thirdl.size(0), -1, 4)
        
        fourthl = nn.Tanh()(self.loc4(fourth))
        fourthl = fourthl.permute(0, 2, 3, 1).contiguous().view(fourthl.size(0), -1, 4)
        
        fifthl = nn.Tanh()(self.loc5(fifth))
        fifthl = fifthl.permute(0, 2, 3, 1).contiguous().view(fifthl.size(0), -1, 4)
        
        sixthl = nn.Tanh()(self.loc6(sixth))
        sixthl = sixthl.permute(0, 2, 3, 1).contiguous().view(sixthl.size(0), -1, 4)
        

        firstc = self.conf1(first)
        firstc = nn.Softmax(dim=-1)(firstc.permute(0, 2, 3, 1).contiguous().view(firstc.size(0), -1,
                                                                                 len(self.classes)))
        
        secondc = self.conf2(second)
        secondc = nn.Softmax(dim=-1)(secondc.permute(0, 2, 3, 1).contiguous().view(secondc.size(0), -1,
                                                                                   len(self.classes)))
        
        thirdc = self.conf3(third)
        thirdc = nn.Softmax(dim=-1)(thirdc.permute(0, 2, 3, 1).contiguous().view(thirdc.size(0), -1,
                                                                                 len(self.classes)))
        
        fourthc = self.conf4(fourth)
        fourthc = nn.Softmax(dim=-1)(fourthc.permute(0, 2, 3, 1).contiguous().view(fourthc.size(0), -1,
                                                                                   len(self.classes)))
        
        fifthc = self.conf5(fifth)
        fifthc = nn.Softmax(dim=-1)(fifthc.permute(0, 2, 3, 1).contiguous().view(fifthc.size(0), -1,
                                                                                 len(self.classes)))
        
        sixthc = self.conf6(sixth)
        sixthc = nn.Softmax(dim=-1)(sixthc.permute(0, 2, 3, 1).contiguous().view(sixthc.size(0), -1,
                                                                                 len(self.classes)))
        
        
        locs = torch.cat((firstl, secondl, thirdl, fourthl, fifthl, sixthl), 1)
        confs = torch.cat((firstc, secondc, thirdc, fourthc, fifthc, sixthc), 1)
        
        return locs, confs


class SSD300(nn.Module):
    def __init__(self):
        super().__init__()
        
        # parameters
        self.input_shape = (None, 3, 300, 300)
        self.aspect_ratios = {
            0: [1, 2, 1 / 2],
            1: [1, 2, 3, 1 / 2, 1 / 3],
            2: [1, 2, 3, 1 / 2, 1 / 3],
            3: [1, 2, 3, 1 / 2, 1 / 3],
            4: [1, 2, 1 / 2],
            5: [1, 2, 1 / 2],
        }
        self.sizes = {
            0: [38, 38],
            1: [19, 19],
            2: [10, 10],
            3: [5, 5],
            4: [3, 3],
            5: [1, 1],
        }
        self.convolution_filters = {
            0: 32,
            1: 96,
            2: 1280,
            3: 256,
            4: 256,
            5: 128,
        }
        self.classes = ['head', 'background']
        
        # load models
        self.backbone = BackBone()
        self.auxbone = AuxiliaryBone(convolution_filters=self.convolution_filters)
        self.predictbone = PredictionBone(aspect_ratios=self.aspect_ratios, sizes=self.sizes,
                                          convolution_filters=self.convolution_filters, classes=self.classes)

    # def __detect(self, locations, confidence):
    #     print(locations.shape[:2], confidence.shape[:2])
    #     assert locations.shape[:2] == confidence.shape[:2]
    #     boxes = []
    #     for element in range(locations.shape[0]):
    #         element_locations = locations[element]
    #         for index_ratio in range(len(self.aspect_ratios[element])):
    #             pass
    #             # for i in
    #     return boxes, confidence
        
    def forward(self, x):
        assert self.input_shape[1:] == x.shape[1:]
        # first (38, 38); second (19, 19); third (10, 10) outs
        first, second, third = self.backbone(x)
        # fourth (5, 5); fifth (3, 3); sixth (1, 1) outs
        fourth, fifth, sixth = self.auxbone(third)
        # return locations and confs N, 9700, 4 and N, 9700, len(classes)
        locs, confs = self.predictbone(first, second, third, fourth, fifth, sixth)
        # boxes, probabilities = self.__detect(locs, confs)
        return locs, confs