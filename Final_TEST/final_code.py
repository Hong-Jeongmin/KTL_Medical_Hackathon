from final_utils import *
from torch.utils.data import DataLoader
import torch, os
from dataset import get_cardiomegaly, cardiomegaly_dataset, get_mammography, mammography_dataset, get_pneumothorax, pneumothorax_dataset, make_pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# CM
print('Start CM')
if not os.path.exists("../Result/CM"):
    os.mkdir("../Result/CM")
    
dataset = 'CM'
image_dim = 512

print('Preprocessing...')
make_pickle(dataset, image_dim)

file, x = get_cardiomegaly(image_dim)
le = LabelEncoder()
file = le.fit_transform(file.ravel())
CM_dataset = cardiomegaly_dataset(file, x)

path = './CM.pth'
model = Inception().cuda()
model.load_state_dict(torch.load(path))
test_loader = DataLoader(CM_dataset, batch_size=8, shuffle=False)
classification(dataset, model, test_loader, le, TTA=True)
torch.cuda.empty_cache()

# PT
print('Start PT')
if not os.path.exists("../Result/PT"):
    os.mkdir("../Result/PT")

dataset = 'PT'
image_dim = 512

print('Preprocessing...')
make_pickle(dataset, image_dim)

file, x, img_size= get_pneumothorax(1024)
le = LabelEncoder()
file = le.fit_transform(file.ravel())
PT_dataset = pneumothorax_dataset(file, x, img_size)

path = './PT_clf.pth'
model = Inception().cuda()
model.load_state_dict(torch.load(path))
test_loader = DataLoader(PT_dataset, batch_size=4, shuffle=False)
classification(dataset, model, test_loader, le, TTA=True)
torch.cuda.empty_cache()

print("PT segmentation")
file, x, img_size= get_pneumothorax(512)
file = le.transform(file.ravel())
df = pd.read_csv('../Result/PT/PT.csv', header=None)
idx = df[1] == 1
PT_dataset = pneumothorax_dataset(file[idx], x[idx], img_size[idx], clf=False)
path = './PT_seg.pth'
model = NestedUNet(num_classes = 1, deep_supervision=False).cuda()
model.load_state_dict(torch.load(path), strict=False)
test_loader = DataLoader(PT_dataset, batch_size=4, shuffle=False)
segmentation(model, test_loader, le)
torch.cuda.empty_cache()


# MG
print('Start MG')
if not os.path.exists("../Result/MG"):
    os.mkdir("../Result/MG")

dataset = 'MG'
image_dim = 512

print('Preprocessing...')
make_pickle(dataset, image_dim)

file, x, img_size = get_mammography(image_dim)
le = LabelEncoder()
file = le.fit_transform(file.ravel())
test_dataset = mammography_dataset(file, x, img_size)

path = './MG.pth'
model = SEInception3().cuda()
model.load_state_dict(torch.load(path))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
classification_MG(model, test_loader, le, TTA=True)
torch.cuda.empty_cache()
