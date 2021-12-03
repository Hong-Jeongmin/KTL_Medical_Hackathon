import os, pickle, sys, pydicom
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
from PIL import Image
from tqdm import tqdm

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


from collections import defaultdict
from torchvision.models import inception_v3


def classification(dataset, model, test_loader, encoder, TTA=False):
    model.eval()
    y_pred = []
    y_hard = []
    files = []
    
    for i in range(30):
        y_pred_one = []
        for file, x, _ in tqdm(test_loader, leave=False):
            if i==0:
                file = encoder.inverse_transform(file)
                files.append(file)
            x = x.cuda()
            y_hat = model(x)
            y_pred_one.append(np.array(y_hat.detach().cpu()).reshape(-1,))
        y_pred.append(np.hstack(y_pred_one))
        if not TTA:
            break
    files = np.hstack(files)
    y_pred = np.mean(y_pred, axis=0)
    y_hard = (y_pred>0.5)+0
    with open(f'../Result/{dataset}/{dataset}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(files, y_hard))

def segmentation(model, test_loader, encoder):
    model.eval()
    for file, x, img_size in test_loader:
        x = x.cuda()
        output = model(x)
        output = torch.sigmoid(output).data.cpu().numpy()

        output_ = (output > 0.5)
        file = encoder.inverse_transform(file)
        for f, m, s in zip(file, output_, img_size):
            m = Image.fromarray((m.squeeze() * 255).astype(np.uint8))
            m = m.resize((s[1], s[0]), Image.LANCZOS)
            m.save(f'../Result/PT/{f}.png')          

def make_prediction(model, img):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) : 
        idx_list = []

        if len(preds[id]['scores']) == 0:
            continue
        idx_list.append(preds[id]['scores'].argmax())
        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list] 
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    
    return preds

def classification_MG(model, test_loader, encoder, TTA=False):
    y_pred = []
    files = []
    
    for i in range(30):
        y_pred_one = []
        for file, x, _ in tqdm(test_loader, leave=False):
            if i==0:
                file = encoder.inverse_transform(file)
                files.append(file)
            x = x.cuda()
            y_hat = model(x)
            y_pred_one.append(np.array(y_hat.detach().cpu()).reshape(-1,))
        y_pred.append(np.hstack(y_pred_one))
        if not TTA:
            break
    files = np.hstack(files)
    y_pred = np.mean(y_pred, axis=0)
    patient = defaultdict(float)
    
    for f, y_hat in zip(files, y_pred):
        name = '_'.join(f.split('_')[:-1])
        patient[name] += y_hat
    
    for k, v in patient.items():
        if patient[k] > 0.6:
            patient[k] = 1
        else:
            patient[k] = 0

    df = pd.DataFrame.from_dict(data = patient, orient='index')
    df.to_csv('../Result/MG/MG.csv', index=True, header=False, mode='w')

    return patient


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.f=inception_v3(aux_logits=False, init_weights=True)
        self.f.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())


    def forward(self, x):
        x = self.f(x)
        return x

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        
        nb_filter = [32,64,128,256,512]
        
        self.deep_supervision = deep_supervision
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1= VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final1(x0_2)
            output3 = self.final1(x0_3)
            output4 = self.final1(x0_4)
            return [output1, output2, output3, output4]
        
        else:
            output = self.final(x0_4)
            return output

class SELayer(nn.Module):
    def __init__(self, channel=3, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class SEInception3(nn.Module):
    def __init__(self, aux_logits=False):
        super(SEInception3, self).__init__()
        model = Inception()
        model.f.Mixed_5b.add_module("SELayer", SELayer(192))
        model.f.Mixed_5c.add_module("SELayer", SELayer(256))
        model.f.Mixed_5d.add_module("SELayer", SELayer(288))
        model.f.Mixed_6a.add_module("SELayer", SELayer(288))
        model.f.Mixed_6b.add_module("SELayer", SELayer(768))
        model.f.Mixed_6c.add_module("SELayer", SELayer(768))
        model.f.Mixed_6d.add_module("SELayer", SELayer(768))
        model.f.Mixed_6e.add_module("SELayer", SELayer(768))
        if aux_logits:
            model.AuxLogits.add_module("SELayer", SELayer(768))
        model.f.Mixed_7a.add_module("SELayer", SELayer(768))
        model.f.Mixed_7b.add_module("SELayer", SELayer(1280))
        model.f.Mixed_7c.add_module("SELayer", SELayer(2048))
        self.model = model
    def forward(self, x):
        return self.model(x)

