import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=7, min_lr=1e-6, verbose=False, delta=0, path='checkpoint.pt', scheduler=None):
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.scheduler = scheduler

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            print(f'Counter {self.counter} --> {self.counter}')
            self.counter += 1
            if self.counter >= self.patience:
                if self.scheduler._last_lr >= self.min_lr:
                    self.scheduler.step()
                    print('Decreased learning rate to', self.scheduler._last_lr)
                    self.counter = 0
                else:
                    self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Savaing model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count 
        
def iou_score(output, target):
    smooth = 1e-5
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    return (intersection + smooth) / (union + smooth)

class DiceCoef(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth=smooth

    def forward(self, input, target):
        input_sigm = torch.sigmoid(input)
        iflat = input_sigm.view(-1)
        tflat = target.view(-1)
        intersection = (iflat*tflat).sum()
        return ((2.0*intersection+self.smooth)/(iflat.sum()+tflat.sum()+self.smooth))

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, smooth):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_coef = DiceCoef(smooth)

    def forward(self, input, target):
        loss = self.alpha*self.bce(input, target) - torch.log(self.dice_coef(input, target))
        return loss.mean()