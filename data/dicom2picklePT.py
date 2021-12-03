import numpy as np
import sys, os, pickle
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from skimage import exposure
from tqdm import tqdm

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
    dpath_original = './CXR_NM'
    dpath_png = dpath_original + '_PT_png'
else:
    dpath_original = './CXR_PT'
    dpath_png = dpath_original + '_png'


# save as png file
if not(os.path.isdir(dpath_png)):
    os.mkdir(dpath_png)

file_list = os.listdir(dpath_original)

# for f in tqdm(file_list):
#     if f.endswith('.dcm'):
#         impath = os.path.join(dpath_original, f)
#         ds = pydicom.read_file(impath)
#         img = ds.pixel_array
#         # try:
#         #     img = apply_voi_lut(img, ds)
#         # except ValueError:
#         #     print('file:', f)
#         #     max_v = np.percentile(img, 95)
#         #     if max_v / np.max(img) < 0.6:
#         #         img = np.clip(img, 0, max_v)
#         img = (img - np.min(img)) / (np.max(img) - np.min(img)) # MIN MAX Scaling
#         max_v = np.percentile(img, 95)
#         if max_v / np.max(img) < 0.6:
#             img = np.clip(img, 0, max_v)
#         h, w = img.shape
#         m = min(h, w)
#         img = np.clip(img, np.percentile(img, 3), np.percentile(img, 97))
#         img = exposure.equalize_adapthist(img, clip_limit=0.005)
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
            
X = []
Y = []
file_list = os.listdir(dpath_png)
for f in file_list:
    img = Image.open(os.path.join(dpath_png, f))
    img_resized = img.resize((image_dim, image_dim), Image.LANCZOS)
    img_resized.dtype=np.uint8
    img_resized = np.array(img_resized, dtype=np.uint8)
    X.append(img_resized)

    if normal:
        mask = np.zeros((image_dim, image_dim), dtype=np.uint8)
        Y.append(mask)
    else:
        mask = Image.open(os.path.join(dpath_original, f)[:-4]+'.png')
        mask_resized = mask.resize((image_dim, image_dim), Image.LANCZOS)
        mask_resized.dtype=np.uint8
        mask_resized = np.array(mask_resized, dtype=np.uint8)
        Y.append(mask_resized)
        
    if len(X) % 100 == 0:
        print(len(X), 'done')

X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

print('X.shape: {} Y.shape: {}'.format(X.shape, Y.shape))
print('X.dtype: {} Y.dtype: {}'.format(X.dtype, Y.dtype))
print('max(X): {} min(X): {}'.format(np.max(X), np.min(X)))

if normal:
    with open(dpath_original+'_PT_'+str(image_dim)+'.pickle', 'wb') as f:
        pickle.dump([X, Y], f)
else:
    with open(dpath_original+'_'+str(image_dim)+'.pickle', 'wb') as f:
        pickle.dump([X, Y], f)
