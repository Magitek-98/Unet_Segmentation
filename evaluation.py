import torch
# SR : Segmentation Result
# GT : Ground Truth

def get_jaccard(y_pred,y_true,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union =  (y_pred + y_true).sum(dim=-2).sum(dim=-1)
    return list((intersection / (union - intersection + epsilon)).data.cpu().numpy())


def get_dice(y_pred,y_true,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = (y_pred + y_true).sum(dim=-2).sum(dim=-1)
    return list(((2. * intersection) / (union + epsilon)).data.cpu().numpy())


def get_sen(y_pred,y_true,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    TP = (((y_pred == 1) + (y_true==1))==2).float().sum(dim=-2).sum(dim=-1)
    FN = (((y_pred == 0) + (y_true==1))==2).float().sum(dim=-2).sum(dim=-1)
    return list((TP/ (TP + FN + epsilon)).data.cpu().numpy())


def get_spe(y_pred,y_true,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    TN = (((y_pred==0) + (y_true==0))==2).float().sum(dim=-2).sum(dim=-1)
    FP = (((y_pred==1) + (y_true==0))==2).float().sum(dim=-2).sum(dim=-1)
    return list((TN / (TN + FP + epsilon)).data.cpu().numpy())


def get_ppv(y_pred,y_true,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    TP = (((y_pred == 1) + (y_true==1))==2).float().sum(dim=-2).sum(dim=-1)
    FP = (((y_pred==1) + (y_true==0))==2).float().sum(dim=-2).sum(dim=-1)
    return list((TP / (TP + FP + epsilon)).data.cpu().numpy())





