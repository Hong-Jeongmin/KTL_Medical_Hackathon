import numpy as np
import sys, os, pickle
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from skimage import exposure
from tqdm import tqdm
from collections import defaultdict

image_dim = int(sys.argv[1]) #224, 299, 331, ...
normal = True if sys.argv[2] == 'normal' else False
print('image dim:', image_dim)
print('normal:', normal)

DATASET_DIR = {'cardiomegaly_abnormal':'./CXR_CM', 'cardiomegaly_normal':'./CXR_NM',
               'pneumothorax_abnormal':'./CXR_PT', 'pneumothorax_normal':'./CXR_NM',
               'mammography_abnormal':'./MG_AN', 'mammography_normal':'./MG_NM'}
DATASET_LEN = {'cardiomegaly_abnormal':1027, 'cardiomegaly_normal':1400,
               'pneumothorax_abnormal':744, 'pneumothorax_normal':1400,
               'mammography_abnormal':564, 'mammography_normal':665}

# change dir
if normal:
    dpath_original = './MG_NM'
else:
    dpath_original = './MG_AN'

dpath_png = dpath_original + '_voi_lut'

# save as png file
if not(os.path.isdir(dpath_png)):
    os.mkdir(dpath_png)

file_list = os.listdir(dpath_original)

# for f in tqdm(file_list):
#     if f.endswith('.dcm'):
#         impath = os.path.join(dpath_original, f)
#         ds = pydicom.read_file(impath)
#         img = ds.pixel_array
#         img = apply_voi_lut(img, ds)
#         img = (img - np.min(img)) / (np.max(img) - np.min(img)) # MIN MAX Scaling
#         # if np.histogram(img.ravel())[0][2] < 5e4:
#         #     hist = np.histogram(img.ravel(), 100)
#         #     min_idx = 50
#         #     while hist[0][min_idx] < 5000:
#         #         min_idx += 1
#         #     img = np.clip(img, a_min=hist[1][min_idx], a_max=np.percentile(img, 99))
#         #     img = (img - np.min(img)) / (np.max(img) - np.min(img)) # MIN MAX Scaling
#         # else:
#         #     continue

#         h, w = img.shape
#         m = min(h, w)
#         img = exposure.equalize_adapthist(img, clip_limit=0.005)
#         # img = exposure.equalize_hist(img)
#         img = (np.maximum(img, 0) / img.max()) * 255.0
#         img = np.uint8(img)
#         if ds.PhotometricInterpretation == 'MONOCHROME2':
#             pass
#         else:
#             img = np.invert(img)
#         img = Image.fromarray(img)
#         img = img.resize((m, m), Image.LANCZOS)
#         filename = os.path.basename(f).strip()[:-4]
#         out_path = os.path.join(dpath_png, filename + '.png')
#         img.save(out_path)

# print('Done transform to png')


files = []
X = []
cords = []
Y = []

if normal:
    file_list = os.listdir(dpath_png)

    for f in file_list:
        img = Image.open(os.path.join(dpath_png, f))
        img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
        img_resized.dtype=np.uint8
        img_resized = np.array(img_resized, dtype=np.uint8)

        files.append(f)
        X.append(img_resized)
        cords.append((0,0,0,0))
        Y.append(0)
            
        if len(X) % 100 == 0:
            print(len(X), 'done')

else:
    file_list = os.listdir(dpath_original)
    file_list = [f for f in file_list if f.endswith('.dcm')]

    for f in file_list:
        img = Image.open(os.path.join(dpath_png, f[:-4]+'.png'))
        img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
        img_resized.dtype=np.uint8
        img_resized = np.array(img_resized, dtype=np.uint8)

        files.append(f)
        X.append(img_resized)

        if os.path.isfile(os.path.join(dpath_original, f[:-4]+'.png')):
            box = Image.open(os.path.join(dpath_original, f[:-4]+'.png'))
            box = box.resize((image_dim, image_dim), Image.LANCZOS)
            box = np.array(box, dtype=np.uint8)
            y, x = np.where(box == 255)
            x1, y1 = x.min(), y.min()
            x2, y2 = x.max(), y.max()
            cords.append((x1, y1, x2, y2))
            Y.append(1)
        else:
            cords.append((0,0,0,0))
            Y.append(0)
 
        if len(X) % 100 == 0:
            print(len(X), 'done')

X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

print('X.shape: {} Y.shape: {}'.format(X.shape, Y.shape))
print('X.dtype: {} Y.dtype: {}'.format(X.dtype, Y.dtype))
print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

with open(dpath_original+'_'+str(image_dim)+'.pickle', 'wb') as f:
    pickle.dump([files, X, cords, Y], f)
