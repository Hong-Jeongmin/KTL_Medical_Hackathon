import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
#from PIL import Image
import pickle
import os
import pydicom

def get_cardiomegaly(image_dim, split=False):
    with open('CXR_CM_'+str(image_dim)+'.pickle', 'rb') as f:
        [X_abnormal, Y_abnormal] = pickle.load(f)
    with open('CXR_NM_'+str(image_dim)+'.pickle', 'rb') as f:
        [X_normal, Y_normal] = pickle.load(f)
    X = np.concatenate((X_abnormal, X_normal))
    Y = np.concatenate((Y_abnormal, Y_normal))

    if split:
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.05, random_state=27407, stratify=Y)
        return X_tr, torch.LongTensor(Y_tr.squeeze()), X_te, torch.LongTensor(Y_te.squeeze())
    else :
        return X, torch.LongTensor(Y.squeeze())

def get_pneumothorax(image_dim, split=False):
    with open('CXR_PT_'+str(image_dim)+'_manual.pickle', 'rb') as f:
        [X_abnormal, mask_abnormal] = pickle.load(f)
    with open('CXR_NM_PT_'+str(image_dim)+'_manual.pickle', 'rb') as f:
        [X_normal, mask_normal] = pickle.load(f)

    X = np.concatenate((X_abnormal, X_normal))
    mask = np.concatenate((mask_abnormal, mask_normal))
    Y = np.concatenate((np.ones(len(mask_abnormal)), np.zeros(len(mask_normal))))

    X_idx = np.arange(len(X_abnormal))
    normal_idx = np.arange(len(X_normal))

    if split:
        X_tr, X_te, mask_tr, mask_te, Y_tr, Y_te = train_test_split(X, mask, Y, test_size=0.05, random_state=2740, stratify=Y)
        return X_tr, mask_tr, torch.LongTensor(Y_tr.squeeze()), X_te, mask_te, torch.LongTensor(Y_te.squeeze())
    else :
        return X, mask, torch.LongTensor(Y.squeeze())

def get_abnormal_PM(image_dim, split=False):
    with open('CXR_PT_'+str(image_dim)+'.pickle', 'rb') as f:
        [X_abnormal, mask_abnormal] = pickle.load(f)
    Y = np.ones(len(mask_abnormal))
    
    if split:
        X_tr, X_te, mask_tr, mask_te, Y_tr, Y_te = train_test_split(X_abnormal, mask_abnormal,Y,  test_size=0.05, random_state=2740)
        return X_tr, mask_tr, torch.LongTensor(Y_tr.squeeze()) ,X_te, mask_te, torch.LongTensor(Y_te.squeeze())
    else :
        return X_abnormal, mask_abnormal, torch.LongTensor(Y.squeeze())

def get_mammography(image_dim, split=False):
    with open('MG_AN_'+str(image_dim)+'_manual.pickle', 'rb') as f:
        [files_abnormal, X_abnormal, cords_abnormal, Y_abnormal] = pickle.load(f)
    with open('MG_NM_'+str(image_dim)+'_manual.pickle', 'rb') as f:
        [files_normal, X_normal, cords_normal, Y_normal] = pickle.load(f)

    # if detection:
    #     idx = [i for i, c in enumerate(cords_abnormal) if c != (0,0,0,0)]
        
    #     cords = np.array(cords_abnormal)[idx]
    #     X = np.array(X_abnormal)[idx]
    #     Y = np.array(Y_abnormal)[idx]

    #     if split:
    #         X_tr, X_te, cords_tr, cords_te, Y_tr, Y_te = train_test_split(X, cords, Y, test_size=0.05, random_state=27407, stratify=Y)
    #         return X_tr, cords_tr, torch.LongTensor(Y_tr.squeeze()), X_te, cords_te, torch.LongTensor(Y_te.squeeze())
    #     else:
    #         return X, cords, Y

    # else: #classification
    files = np.concatenate((files_abnormal, files_normal))
    X = np.concatenate((X_abnormal, X_normal))
    Y = np.concatenate((Y_abnormal, Y_normal))

    if split:
        files_tr, files_te, X_tr, X_te, Y_tr, Y_te = train_test_split(files, X, Y, test_size=0.25, random_state=27407, stratify=Y)
        return X_tr, torch.LongTensor(Y_tr.squeeze()), X_te, torch.LongTensor(Y_te.squeeze())
    else:
        return X, Y


class mammography_rcnn_dataset(Dataset):
    def __init__(self, X, cords, Y, transforms):
        self.X = X
        self.cords = cords
        self.Y = Y
        self.transforms = transforms

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)

        xmin, ymin, xmax, ymax = self.cords[index]
        boxes = [[xmin, ymin, xmax, ymax]]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.LongTensor((y,))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is None:
            t = transforms.Compose([
                transforms.ToTensor()
                ])
            img = t(x)
        else:
            img = self.transforms(x)

        return img, target

    def __len__(self):
        return len(self.Y)

class mammography_clf_dataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)
        if self.transform is None:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0), (1))
                ])
            x = t(x)
        else:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

class cardiomegaly_dataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)
        if self.transform is None:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.6912), (0.1859))
                ])
            x = t(x)
        else:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

class pneumothorax_dataset(Dataset):
    def __init__(self, X, mask, Y, transform=None, clf=False):
        self.X = X
        self.mask = mask
        self.Y = Y
        self.transform = transform
        self.clf = clf

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1) # 1ch to 3ch
        

        if self.transform is None:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.6912), (0.1859))
                ])
            x = t(x)
        else:
            x = self.transform(x)

        if self.clf:
            return x, y
        else:
            mask = self.mask[index]
            tensorize = transforms.ToTensor()
            mask = tensorize(mask)

            p = torch.rand(1).item()
            if p > 0.5:
                x = transforms.functional.hflip(x)
                mask = transforms.functional.hflip(mask)
            p = torch.rand(1).item()
            if p > 0.5:
                sigma = torch.empty(1).uniform_(0.1, 2.0).item()
                x = transforms.functional.gaussian_blur(x, 3, [sigma, sigma])
            return x, mask, y

    def __len__(self):
        return len(self.X)