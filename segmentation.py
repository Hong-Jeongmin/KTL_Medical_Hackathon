import os
import argparse
import csv, json
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from model import UNet, NestedUNet, ResUnet
from data.dataset import get_pneumothorax, pneumothorax_dataset, get_abnormal_PM
from utils import AverageMeter, iou_score, BCEDiceLoss
from collections import OrderedDict

def train(train_lodaer, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou':AverageMeter()}
    
    model.train()
    
    pbar = tqdm(total = len(train_loader), leave=False)
    for input, target, y in train_loader:
        input, target = input.cuda(), target.cuda()
        mask = y.bool()
        
        if deep_supervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1][mask], target[mask])
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output[mask], target[mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        
        postfix = OrderedDict([('loss', avg_meters['loss'].avg),
                              ('iou', avg_meters['iou'].avg),
                              ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                       ('iou',avg_meters['iou'].avg)])

def validate(test_loader, model, criterion):
    avg_meters = {'loss':AverageMeter(), 'iou':AverageMeter()}
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), leave=False)
        for input, target, y in test_loader:
            input, target = input.cuda(), target.cuda()
            mask = y.bool()

            if deep_supervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1][mask], target[mask])
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output[mask], target[mask])

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                      ('iou', avg_meters['iou'].avg)])

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--model', default='unet', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--decay_factor', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--deep_supervision', default=False, action='store_true')

    P = parser.parse_args()

    ##############should modify######
    # prefix = 'Dice'+P.model+f'+{P.dim}_{P.batch_size}'
    prefix = 'Total_dataset'+P.model+f'+FLIP_BLUR+{P.dim}_{P.batch_size}'

    X_train, mask_train, Y_train, X_test, mask_test, Y_test= get_pneumothorax(image_dim=P.dim ,split=True)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(5),
        transforms.ColorJitter(contrast=0.2, brightness=0.2),
        transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
    ])

    if P.model == 'unet':
        model = UNet(num_classes = 1).cuda()
        deep_supervision = P.deep_supervision
    elif P.model == 'nestedunet':
        model = NestedUNet(num_classes = 1, deep_supervision=P.deep_supervision).cuda()
        deep_supervision = P.deep_supervision
    elif P.model == 'resunet':
        model = ResUnet(num_classes = 1, deep_supervision=P.deep_supervision).cuda()
        deep_supervision = P.deep_supervision
    
    optimizer = optim.Adam(model.parameters(), lr=P.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=P.decay_factor, patience=P.patience, min_lr=P.min_lr, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = BCEDiceLoss(1,1)
    train_dataset = pneumothorax_dataset(X_train, mask_train, Y_train, transform = train_transform)
    test_dataset = pneumothorax_dataset(X_test, mask_test, Y_test, transform = test_transform)
    train_loader = DataLoader(train_dataset, batch_size = P.batch_size, num_workers= 4, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = P.batch_size, num_workers = 4, shuffle = False)
    print('Training...')
    train_result = {'train_loss' :[], 'accuracy':[],'auroc':[]}
    best_iou, best_loss = 0, np.Inf
    
    log = OrderedDict([
        ('epoch',[]),
        ('lr',[]),
        ('loss',[]),
        ('iou',[]),
        ('val_loss',[]),
        ('val_iou',[]),
    ])

    for epoch in range(1, P.epochs+1):
        print('Epoch [%d/%d]'%(epoch, P.epochs))
        
        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(test_loader, model, criterion)
        
        scheduler.step(val_log['loss'])
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
             %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        
        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'results/{prefix}_best_iou.pth')
            best_iou = val_log['iou']
        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), f'results/{prefix}_best_loss.pth')
            best_iou = val_log['loss']
        if scheduler._last_lr[0] <= P.min_lr + 1e-10 and scheduler.num_bad_epochs >= scheduler.patience:
            break
#         torch.cuda.empty_cache()

    with open('results/'+prefix+f'_train_loss.json', 'w') as f:
        json.dump(log, f)

    # best iou
    model.eval()
    model.load_state_dict(torch.load(f'results/{prefix}_best_iou.pth'))
    val_log = validate(test_loader, model, criterion)
    with open('results/'+prefix+f'_best_iou.json', 'w') as f:
        json.dump(log, f)

    # best loss
    model.eval()
    model.load_state_dict(torch.load(f'results/{prefix}_best_loss.pth'))
    val_log = validate(test_loader, model, criterion)
    with open('results/'+prefix+f'_best_loss.json', 'w') as f:
        json.dump(log, f)