import torch
import torch.nn.functional as F

# def rmse_loss(pred, target):
#     labels_tensor = torch.tensor([float(x) for x in target], dtype=torch.float32).unsqueeze(1).to(pred.device)
#     return F.mse_loss(pred, labels_tensor )
def rmse_loss(pred, target):
    labels_tensor = torch.tensor([float(x) for x in target], dtype=torch.float32).unsqueeze(1).to(pred.device)
    return  torch.sqrt(F.mse_loss(pred, labels_tensor ))

def mae_loss(pred, target):
    # 将目标值转换为浮点数张量，并调整形状以匹配预测值
    labels_tensor = torch.tensor([float(x) for x in target], dtype=torch.float32).unsqueeze(1).to(pred.device)
    # 计算绝对误差
    absolute_errors = torch.abs(pred - labels_tensor)
    # 返回平均绝对误差
    return torch.mean(absolute_errors)