# coding: utf-8

# # Import

from torch import nn
from torchvision.models import mobilenet_v2
import torch
import numpy as np
import matplotlib.pyplot as plt


# # Utils

class Utils():
    def __init__(self):
        pass

    @staticmethod
    def IoU(first, second):
        l1, r1 = first[:2], first[2:]
        l2, r2 = second[:2], second[2:]

        x = 0
        y = 1

        # Area of 1st Rectangle
        area1 = abs(l1[x] - r1[x]) * abs(l1[y] - r1[y])

        # Area of 2nd Rectangle
        area2 = abs(l2[x] - r2[x]) * abs(l2[y] - r2[y])

        x_dist = (min(r1[x], r2[x]) -
                  max(l1[x], l2[x]))

        y_dist = (min(r1[y], r2[y]) -
                  max(l1[y], l2[y]))
        areaI = 0
        if x_dist > 0 and y_dist > 0:
            areaI = x_dist * y_dist
        return areaI / (area1 + area2 - areaI)


# # Loss

class Multibox_Loss(nn.Module):
    def __init__(self, alpha):
        super(Multibox_Loss, self).__init__()

        self.alpha = alpha
        self.utils = Utils()

    def forward(self, predict_boxes, predict_probabilities, true_boxes, true_probabilities):
        print(len(predict_boxes), len(predict_probabilities), len(true_boxes), len(true_probabilities))
        assert len(predict_boxes) == len(predict_probabilities) == len(true_boxes) == len(true_probabilities)
        batch = len(predict_boxes)
        res_loss = 0
        for i in range(batch):
            # print(predict_boxes[i].shape, predict_probabilities[i].shape, true_boxes[i].shape, true_probabilities[i].shape)
            for j in range(len(predict_boxes[i])):
                for l in range(len(true_boxes[i])):
                    x1, y1, x2, y2 = predict_boxes[i][j]
                    x3, y3, x4, y4 = true_boxes[i][l]
                    intersection_of_union = self.utils.IoU((x1, y1, x2, y2), (x3, y3, x4, y4))
                    if intersection_of_union < 0.5:
                        res_loss += abs(predict_probabilities[i][j] - true_probabilities[i][l]) ** 0.1
                    else:
                        # print('Found', predict_boxes[i][j])
                        res_loss += ((predict_boxes[i][j][0] + predict_boxes[i][j][2]) * 150 - (true_boxes[i][l][0] + true_boxes[i][l][2]) * 150) ** 2 * self.alpha
                        res_loss += ((predict_boxes[i][j][1] + predict_boxes[i][j][3]) * 150 - (true_boxes[i][l][1] + true_boxes[i][l][3]) * 150) ** 2 * self.alpha
                        res_loss += (abs(abs(predict_boxes[i][j][0] - predict_boxes[i][j][2]) - abs(true_boxes[i][l][0] - true_boxes[i][l][2])) * 300) ** 2 * self.alpha
                        res_loss += (abs(abs(predict_boxes[i][j][1] - predict_boxes[i][j][3]) - abs(true_boxes[i][l][1] - true_boxes[i][l][3])) * 300) ** 2 * self.alpha
        return res_loss


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
        self.scales = {
            0: 0.1,
            1: 0.2,
            2: 0.375,
            3: 0.55,
            4: 0.725,
            5: 0.9,
        }
        self.classes = ['head', 'background']
        
        # load models
        self.backbone = BackBone()
        self.auxbone = AuxiliaryBone(convolution_filters=self.convolution_filters)
        self.predictbone = PredictionBone(aspect_ratios=self.aspect_ratios, sizes=self.sizes,
                                          convolution_filters=self.convolution_filters, classes=self.classes)

    def detect(self, locations, probabilities):
        assert locations.shape[:2] == probabilities.shape[:2]
        len_batch = (locations.shape[0] + probabilities.shape[0]) // 2
        res_boxes = []
        res_probabilities = []
        res_labels = []
        probabilities, labels = probabilities.max(dim=-1)
        # print(locations.shape, probabilities.shape, labels.shape)
        for index in range(len_batch):
            element_locations = locations[index]
            element_probabilities = probabilities[index]
            element_labels = labels[index]
            res_element_boxes = []
            res_element_probabilities = []
            res_element_labels = []
            for jndex in range(6):
                height, width = self.sizes[jndex]
                size = len(self.aspect_ratios[jndex]) * height * width
                channels_locations = len(self.aspect_ratios[jndex]) * 4
                channels_probabilities = len(self.aspect_ratios[jndex])
                channels_labels = channels_probabilities
                congress_locations = element_locations[:size].reshape(height, width, channels_locations)
                congress_probabilities = element_probabilities[:size].reshape(height, width, channels_probabilities)
                congress_labels = element_labels[:size].reshape(height, width, channels_labels)
                element_locations = element_locations[size:]
                element_probabilities = element_probabilities[size:]
                element_labels = element_labels[size:]
                # print(congress_locations.shape)
                for i in range(height):
                    for j in range(width):
                        frst, snd, thrd, frth = congress_locations[i][j][:channels_locations // 4], \
                                     congress_locations[i][j][channels_locations // 4:channels_locations // 4 * 2],\
                                     congress_locations[i][j][channels_locations // 4 * 2:channels_locations // 4 * 3],\
                                     congress_locations[i][j][channels_locations // 4 * 3:channels_locations // 4 * 4]
                        packs = tuple(zip(frst, snd, thrd, frth, congress_labels[i][j]))
                        for andex in range(len(self.aspect_ratios[jndex])):
                            x, y, w, h, cls = packs[andex]
                            if cls == 0:
                                continue
                            # print(x, y, w, h, cls)
                            cx = (j + 0.5 + x) / width
                            cy = (i + 0.5 + y) / height
                            cw = self.scales[jndex] * (self.aspect_ratios[jndex][andex]) ** 0.5
                            ch = self.scales[jndex] / (self.aspect_ratios[jndex][andex]) ** 0.5
                            res_element_boxes.append((cx, cy, cw + (self.scales[jndex] / 2 * w), ch + (self.scales[jndex] / 2 * h)))
                            res_element_probabilities.append([congress_probabilities[i][j][andex]])
                            res_element_labels.append([congress_labels[i][j][andex]])
            res_boxes.append(torch.Tensor(res_element_boxes))
            res_probabilities.append(torch.Tensor(res_element_probabilities))
            res_labels.append(torch.Tensor(res_element_labels))

        return res_boxes, res_probabilities, res_labels
        
    def forward(self, x):
        assert self.input_shape[1:] == x.shape[1:]
        # first (38, 38); second (19, 19); third (10, 10) outs
        first, second, third = self.backbone(x)
        # fourth (5, 5); fifth (3, 3); sixth (1, 1) outs
        fourth, fifth, sixth = self.auxbone(third)
        # return locations and confs N, 9700, 4 and N, 9700, len(classes)
        locs, confs = self.predictbone(first, second, third, fourth, fifth, sixth)
        boxes, probabilities, labels = self.detect(locs, confs)
        return boxes, probabilities, labels
        # return locs, confs