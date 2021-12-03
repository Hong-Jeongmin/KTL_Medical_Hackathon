import os
import argparse
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from model import SEInception3, VGG, Resnet, Inception, Densenet
from dataset import get_cardiomegaly, cardiomegaly_dataset, get_pneumothorax, pneumothorax_dataset, get_mammography, mammography_clf_dataset
from Efficientnet import efficientnet_b0, efficientnet_b1

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CM', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--model', default='vgg19', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--dim', default=512, type=int)

    P = parser.parse_args()
 
    if not os.path.isdir('results'):
        os.mkdir('results')

    prefix = 'final_'+P.dataset+'+'+P.model+f'+{P.dim}_{P.batch_size}'

    if P.dataset == 'CM':
        # X_train, Y_train= get_cardiomegaly(image_dim=512 ,split=False)
        X_train, Y_train, X_test, Y_test = get_cardiomegaly(image_dim=512 ,split=True)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5),
            transforms.ColorJitter(contrast=0.2, brightness=0.2),
            transforms.Normalize(mean=[0.665, 0.665, 0.665], std=[0.268, 0.268, 0.268])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.665, 0.665, 0.665], std=[0.268, 0.268, 0.268])
        ])
        
        train_dataset = cardiomegaly_dataset(X_train, Y_train, transform=train_transform)
        test_dataset = cardiomegaly_dataset(X_test, Y_test, transform=test_transform)
    elif P.dataset == 'PT':
        # X_train, mask_train, Y_train = get_pneumothorax(image_dim=P.dim ,split=False)
        X_train, mask_train, Y_train, X_test, mask_test, Y_test = get_pneumothorax(image_dim=P.dim ,split=True)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(contrast=0.2, brightness=0.2),
            transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
        ])
        train_dataset = pneumothorax_dataset(X_train, mask_train, Y_train, transform=train_transform, clf=True)
        test_dataset = pneumothorax_dataset(X_test, mask_test, Y_test, transform=test_transform, clf=True)
    elif P.dataset == 'MG':
        # X_train, Y_train, X_test, Y_test = get_mammography(image_dim=P.dim, split=True)
        X_test, Y_test = get_mammography(image_dim=P.dim, split=False)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.2, brightness=0.2),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # train_dataset = mammography_clf_dataset(X_train, Y_train, transform=train_transform)
        test_dataset = mammography_clf_dataset(X_test, Y_test, transform=test_transform)
    

    if P.model == 'vgg16':
        model = VGG(ver=16).cuda()
    elif P.model == 'vgg19':
        model = VGG(ver=19).cuda()
    elif P.model == 'resnet18':
        model = Resnet(ver=18).cuda()
    elif P.model == 'resnet34':
        model = Resnet(ver=34).cuda()
    elif P.model == 'resnet50':
        model = Resnet(ver=50).cuda()
    elif P.model == 'densnet121':
        model = Densenet(ver=121).cuda()
    elif P.model == 'densenet161':
        model = Densenet(ver=161).cuda()
    elif P.model == 'inception':
        model = Inception().cuda()
    elif P.model == 'efficientnet_b0':
        model = efficientnet_b0().cuda()
    elif P.model == 'efficientnet_b1':
        model = efficientnet_b1().cuda()
    elif P.model == 'seinception':
        model = SEInception3().cuda()
    optimizer = optim.Adam(model.parameters(), lr=P.lr, weight_decay=1e-6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7, min_lr=P.min_lr, verbose=True)

    # train_loader = DataLoader(train_dataset, batch_size=P.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=P.batch_size, num_workers=4, shuffle=True)
    # print('Training...')
    # train_result = {'train_loss' :[], 'accuracy':[],'auroc':[]}
    # best_accuracy, best_auroc, best_loss = 0, 0, np.Inf
    
    # for epoch in range(1, P.epochs+1):
    #     model.train()
    #     total_loss, total_num = 0,0
    #     data_bar = tqdm(train_loader, leave=False)
    #     for x, y in data_bar:
    #         x, y = x.cuda(), y.cuda()
    #         y = y.reshape(-1, 1).float()
    #         y_hat = model(x)
    #         loss = F.binary_cross_entropy(y_hat, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_num += x.shape[0]
    #         total_loss += loss.item()*x.shape[0]

    #         data_bar.set_description('Epoch:{}/{} Loss :{:.4f}'.format(epoch, P.epochs, total_loss/total_num))
    #     train_result['train_loss'].append(total_loss/total_num)
    #     print(f"Epoch : {epoch}/{P.epochs}, Loss : {total_loss/total_num}")
        
    # #evaluation phase
    #     y_pred = []
    #     y_hard = []
    #     y_true = []

    #     model.eval()
    #     with torch.no_grad():
    #         valid_loss, valid_num = 0, 0
    #         for x, y in test_loader:
    #             x = x.cuda()
    #             y_hat = model(x)
    #             loss = F.binary_cross_entropy(y_hat, y.cuda().reshape(-1, 1).float())

    #             y_pred.append(y_hat.detach().cpu().numpy().reshape(-1,))
    #             y_true.append(y.detach().cpu().numpy())

    #             valid_num += x.shape[0]
    #             valid_loss += loss.item()*x.shape[0]

    #         scheduler.step(valid_loss)

    #     y_pred = np.hstack(y_pred)
    #     y_true = np.hstack(y_true)
    #     y_hard = (y_pred>0.5)+0
    #     accuracy = accuracy_score(y_true, y_hard)
    #     auroc = roc_auc_score(y_true, y_pred)
    #     print(f'Epoch : {epoch}, Accuracy : {accuracy:.4f}, AUROC : {auroc:.4f}')

    #     train_result['accuracy'].append(accuracy)
    #     train_result['auroc'].append(auroc)

    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         torch.save(model.state_dict(), f'results/{prefix}_best_accuracy.pth')
    #     if auroc > best_auroc:
    #         best_auroc = auroc
    #         torch.save(model.state_dict(), f'results/{prefix}_best_auroc.pth')
    #     if valid_loss < best_loss:
    #         best_loss = valid_loss
    #         torch.save(model.state_dict(), f'results/{prefix}_best_loss.pth')
        
    #     if scheduler._last_lr[0] <= P.min_lr + 1e-10 and scheduler.num_bad_epochs >= scheduler.patience:
    #         break

    # # data_frame = pd.DataFrame(data=train_result, index=range(1, P.epochs+1))
    # data_frame = pd.DataFrame(data=train_result, index=range(1, scheduler.last_epoch+1))
    # data_frame.to_csv('results/'+prefix+f'_train_loss.csv', index_label='epoch')

    #best accuracy
    print("Test Best Accuracy Model")
    model.eval()
    model.load_state_dict(torch.load(f'results/{prefix}_best_loss.pth'))
    y_pred = []
    y_hard = []
    y_true = []
    for x, y in test_loader:
        x = x.cuda()
        y_hat = model(x)
        y_pred.append(np.array(y_hat.detach().cpu()).reshape(-1,))
        y_true.append(np.array(y.detach().cpu()).reshape(-1,))
    y_pred = np.hstack(y_pred)
    y_true = np.hstack(y_true)
    y_hard = (y_pred>0.5)+0

    
    test_result = {'data':[],'model':[],'dim':[],'accuracy':[],'precision':[], 'recall':[],'confusion':[], 'auroc':[]}
    test_result['data']=P.dataset
    test_result['model'] = P.model
    test_result['dim'] = P.dim
    test_result['accuracy'] = accuracy_score(y_true, y_hard)
    test_result['precision'] = precision_score(y_true, y_hard)
    test_result['recall'] = recall_score(y_true, y_hard)
    test_result['confusion'] = confusion_matrix(y_true, y_hard)
    test_result['auroc'] = roc_auc_score(y_true, y_pred)
    df = pd.DataFrame.from_dict(data = [test_result])
    if not os.path.exists('results/output.csv'):
        df.to_csv('results/output.csv', index=False, mode='w')
    else:
        df.to_csv('results/output.csv', index=False, mode='a', header=False)
    print(test_result)

    # #best auroc
    # print("Test Best AUROC Model")

    # model.load_state_dict(torch.load(f'results/{prefix}_best_auroc.pth'))
    # y_pred = []
    # y_hard = []
    # y_true = []
    # for x, y in test_loader:
    #     x = x.cuda()
    #     y_hat = model(x)
    #     y_pred.append(np.array(y_hat.detach().cpu()).reshape(-1,))
    #     y_true.append(np.array(y.detach().cpu()).reshape(-1,))
    # y_pred = np.hstack(y_pred)
    # y_true = np.hstack(y_true)
    # y_hard = (y_pred>0.5)+0

    # test_result = dict()
    # test_result['data'] = P.dataset
    # test_result['model'] = P.model
    # test_result['dim'] = P.dim
    # test_result['accuracy'] = accuracy_score(y_true, y_hard)
    # test_result['precision'] = precision_score(y_true, y_hard)
    # test_result['recall'] = recall_score(y_true, y_hard)
    # test_result['confusion'] = confusion_matrix(y_true, y_hard)
    # test_result['auroc'] = roc_auc_score(y_true, y_pred)
    # df = pd.DataFrame.from_dict(data = [test_result])
    # if not os.path.exists('results/output.csv'):
    #     df.to_csv('results/output.csv', index=False, mode='w')
    # else:
    #     df.to_csv('results/output.csv', index=False, mode='a', header=False)

    # #best loss
    # print("Test Best Loss Model")
    # model.load_state_dict(torch.load(f'results/{prefix}_best_loss.pth'))
    # y_pred = []
    # y_hard = []
    # y_true = []
    # for x, y in test_loader:
    #     x = x.cuda()
    #     y_hat = model(x)
    #     y_pred.append(np.array(y_hat.detach().cpu()).reshape(-1,))
    #     y_true.append(np.array(y.detach().cpu()).reshape(-1,))
    # y_pred = np.hstack(y_pred)
    # y_true = np.hstack(y_true)
    # y_hard = (y_pred>0.5)+0


    # test_result = dict()
    # test_result['data'] =P.dataset
    # test_result['model'] = P.model
    # test_result['dim'] = P.dim
    # test_result['accuracy'] = accuracy_score(y_true, y_hard)
    # test_result['precision'] = precision_score(y_true, y_hard)
    # test_result['recall'] = recall_score(y_true, y_hard)
    # test_result['confusion'] = confusion_matrix(y_true, y_hard)
    # test_result['auroc'] = roc_auc_score(y_true, y_pred)
    # df = pd.DataFrame.from_dict(data = [test_result])
    # if not os.path.exists('results/output.csv'):
    #     df.to_csv('results/output.csv', index=False, mode='w')
    # else:
    #     df.to_csv('results/output.csv', index=False, mode='a', header=False) 

    
