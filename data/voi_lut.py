import numpy as np
import sys, os, pickle
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from skimage import exposure
from tqdm import tqdm
dataset = str(sys.argv[1])
image_dim = int(sys.argv[2]) #224, 299, 331, ...
normal = True if sys.argv[3] == 'normal' else False
from dataset import read_xray

print('image dim:', image_dim)
print('normal:', normal)

DATASET_DIR = {'CM_ab':'./CXR_CM', 'CM_nor':'./CXR_NM',
               'PT_ab':'./CXR_PT', 'PT_nor':'./CXR_NM',
               'MG_ab':'./MG_AN', 'MG_nor':'./MG_NM'}
DATASET_LEN = {'cardiomegaly_abnormal':1027, 'cardiomegaly_normal':1400,
               'pneumothorax_abnormal':744, 'pneumothorax_normal':1400,
               'mammography_abnormal':564, 'mammography_normal':665}
dpath_original = DATASET_DIR[dataset]

# change dir
if normal:
    label = 0
else:
    label = 1
dpath_png = dpath_original + 'voilut_png'

# save as png file
if not(os.path.isdir(dpath_png)):
    os.mkdir(dpath_png)

file_list = os.listdir(dpath_original)

for f in tqdm(file_list):
    if f.endswith('.dcm'):
        impath = os.path.join(dpath_original, f)
        img = read_xray(impath)
        if dataset.split('_')[0]=='CM':
            h, w = img.shape
            m = min(h, w)
            crop = int(0.05*m)
            # img = img[:m, :m]
            # img = img[h-m:h, w-m:w]
            img = img[h-m:h, w-m:w][crop:m-crop, crop:m-crop]
        img = exposure.equalize_adapthist(img, clip_limit=0.005)
        img = Image.fromarray(img)
        filename = os.path.basename(f).strip()[:-4]
        out_path = os.path.join(dpath_png, filename + '.png')
        img.save(out_path)

print('Done transform to png')
        
X = []
Y = [] # abnormal 1, normal 0
file_list = os.listdir(dpath_png)
for f in file_list:
    img = Image.open(os.path.join(dpath_png, f))
    img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
    # img_resized = resize(img, (image_dim, image_dim), anti_aliasing=True)
    img_resized.dtype=np.uint8
    img_resized = np.array(img_resized, dtype=np.uint8)
    X.append(img_resized)
    Y.append([label])

    if len(X) % 100 == 0:
        print(len(X), 'done')

X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

print('X.shape: {} Y.shape: {}'.format(X.shape, Y.shape))
print('X.dtype: {} Y.dtype: {}'.format(X.dtype, Y.dtype))
print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

with open(dpath_original+'_'+str(image_dim)+'.pickle', 'wb') as f:
    pickle.dump([X, Y], f)
