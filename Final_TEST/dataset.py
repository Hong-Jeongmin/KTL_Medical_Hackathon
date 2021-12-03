import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pickle
import os
import pydicom
from skimage import exposure
from tqdm import tqdm
from PIL import Image

def get_cardiomegaly(image_dim):
    with open('CM_'+str(image_dim)+'.pickle', 'rb') as f:
        [file_name, X] = pickle.load(f)

    return file_name, X

def get_pneumothorax(image_dim):
    with open('PT_'+str(image_dim)+'.pickle', 'rb') as f:
        [file_name, X, img_size] = pickle.load(f)

    return file_name, X, img_size

def get_mammography(image_dim):
    with open('MG_'+str(image_dim)+'.pickle', 'rb') as f:
        [file_name, X, img_size] = pickle.load(f)

    return file_name, X, img_size

class cardiomegaly_dataset(Dataset):
    def __init__(self, file_name, X):
        self.file_name = file_name
        self.X = X

    def __getitem__(self, index):
        f, x = self.file_name[index], self.X[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5),
            transforms.ColorJitter(contrast=0.2, brightness=0.2),
            transforms.Normalize(mean=[0.665, 0.665, 0.665], std=[0.268, 0.268, 0.268])
        ])

        x = transform(x)

        return f, x, 0

    def __len__(self):
        return len(self.X)


class pneumothorax_dataset(Dataset):
    def __init__(self, file_name, X, img_size, clf=True):
        self.file_name = file_name
        self.X = X
        self.img_size = img_size
        self.clf = clf

    def __getitem__(self, index):
        f, x, s = self.file_name[index], self.X[index], self.img_size[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)
        
        if self.clf:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(contrast=0.2, brightness=0.2),
                transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
            ])
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.567, 0.567, 0.567], std=[0.277, 0.277, 0.277])
            ])

        x = test_transform(x)

        return f, x, s

    def __len__(self):
        return len(self.X)

class mammography_dataset(Dataset):
    def __init__(self, file_name, X, img_size):
        self.file_name = file_name
        self.X = X
        self.img_size = img_size

    def __getitem__(self, index):
        f, x, s = self.file_name[index], self.X[index], self.img_size[index]
        x = np.expand_dims(x, -1)
        x = np.concatenate((x, x, x), -1)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.2, brightness=0.2)
        ])

        x = test_transform(x)

        return f, x, s

    def __len__(self):
        return len(self.X)



def make_pickle(dataset, image_dim):
    file_name = []
    X = []

    if dataset == 'CM':
        dpath_original = '../Test/CM'

        file_list = os.listdir(dpath_original)

        for f in tqdm(file_list):
            if f.endswith('.dcm'):
                impath = os.path.join(dpath_original, f)
                ds = pydicom.read_file(impath)
                img = ds.pixel_array

                # clip brightest pixels
                max_v = np.percentile(img, 95)
                if max_v / np.max(img) < 0.6:
                    img = np.clip(img, 0, max_v)
                
                # MINMAX scaling
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

                # crop image
                h, w = img.shape
                m = min(h, w)
                crop = int(0.05*m)
                img = img[h-m:h, w-m:w][crop:m-crop, crop:m-crop]

                # CLAHE
                img = exposure.equalize_adapthist(img, clip_limit=0.005)

                # save as uint8
                img = (np.maximum(img, 0) / img.max()) * 255.0
                img = np.uint8(img)
                if ds.PhotometricInterpretation == 'MONOCHROME2':
                    pass
                else:
                    img = np.invert(img)
                img = Image.fromarray(img)
                img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
                img_resized.dtype=np.uint8
                img_resized = np.array(img_resized, dtype=np.uint8)

                file_name.append([f[:-4]])
                X.append(img_resized)

                if len(X) % 100 == 0:
                    print(len(X), 'done')

        file_name = np.array(file_name)
        X = np.array(X, dtype=np.uint8)

        print('X.shape: {} X.dtype: {}'.format(X.shape, X.dtype))
        print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

        with open(dataset+'_'+str(image_dim)+'.pickle', 'wb') as f:
            pickle.dump([file_name, X], f)

    elif dataset == 'PT':
        X_1024 = []
        img_size = []

        dpath_original = '../Test/PT'

        file_list = os.listdir(dpath_original)

        for f in tqdm(file_list):
            if f.endswith('.dcm'):
                impath = os.path.join(dpath_original, f)
                ds = pydicom.read_file(impath)
                img = ds.pixel_array

                # clip brightest pixels
                max_v = np.percentile(img, 95)
                if max_v / np.max(img) < 0.6:
                    img = np.clip(img, 0, max_v)
                
                # MINMAX scaling
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

                # save img size
                img_size.append(img.shape)

                # CLAHE
                img = exposure.equalize_adapthist(img, clip_limit=0.005)

                # save as uint8
                img = (np.maximum(img, 0) / img.max()) * 255.0
                img = np.uint8(img)
                if ds.PhotometricInterpretation == 'MONOCHROME2':
                    pass
                else:
                    img = np.invert(img)
                img = Image.fromarray(img)
                img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
                img_resized_1024 = img.resize((1024, 1024), Image.LANCZOS)
                img_resized.dtype=np.uint8
                img_resized_1024.dtype=np.uint8
                img_resized = np.array(img_resized, dtype=np.uint8)
                img_resized_1024 = np.array(img_resized_1024, dtype=np.uint8)

                file_name.append([f[:-4]])
                X.append(img_resized)
                X_1024.append(img_resized_1024)

                if len(X) % 100 == 0:
                    print(len(X), 'done')

        file_name = np.array(file_name)
        X = np.array(X, dtype=np.uint8)
        X_1024 = np.array(X_1024, dtype=np.uint8)
        img_size = np.array(img_size)

        print('X.shape: {} X.dtype: {}'.format(X.shape, X.dtype))
        print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

        with open(dataset+'_'+str(image_dim)+'.pickle', 'wb') as f:
            pickle.dump([file_name, X, img_size], f)
        with open(dataset+'_'+str(1024)+'.pickle', 'wb') as f:
            pickle.dump([file_name, X_1024, img_size], f)

    elif dataset == 'MG':
        img_size = []

        dpath_original = '../Test/MG'

        file_list = os.listdir(dpath_original)
        for f in tqdm(file_list):
            if f.endswith('.dcm'):
                impath = os.path.join(dpath_original, f)
                ds = pydicom.read_file(impath)
                img = ds.pixel_array

                if np.histogram(img.ravel())[0][2] < 5e4:
                    hist = np.histogram(img.ravel(), 100)
                    min_idx = 50
                    while hist[0][min_idx] < 5000:
                        min_idx += 1
                        #print(min_idx)
                        #print(len(hist[0]))
                        if(min_idx == len(hist[0])):
                            break
                    img = np.clip(img, hist[1][min_idx], np.percentile(img, 99))
                else:
                    img = np.clip(img, 0, np.percentile(img, 99))
                
                # MINMAX scaling
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

                # save img size
                img_size.append(img.shape)

                # CLAHE
                img = exposure.equalize_adapthist(img, clip_limit=0.005)

                # save as uint8
                img = (np.maximum(img, 0) / img.max()) * 255.0
                img = np.uint8(img)
                if ds.PhotometricInterpretation == 'MONOCHROME2':
                    pass
                else:
                    img = np.invert(img)
                img = Image.fromarray(img)
                img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
                img_resized.dtype=np.uint8
                img_resized = np.array(img_resized, dtype=np.uint8)

                file_name.append([f[:-4]])
                X.append(img_resized)

                if len(X) % 100 == 0:
                    print(len(X), 'done')

        file_name = np.array(file_name)
        X = np.array(X, dtype=np.uint8)
        img_size = np.array(img_size)

        print('X.shape: {} X.dtype: {}'.format(X.shape, X.dtype))
        print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

        with open(dataset+'_'+str(image_dim)+'.pickle', 'wb') as f:
            pickle.dump([file_name, X, img_size], f)
