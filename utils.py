'''
Some util functions
Part of the code is referenced from Kaggle
'''

import os
import logging
import torch
import random
import numpy as np

from opts import fmix
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.cuda.amp import autocast

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def seed_everything(seed):
    '''固定各类随机种子，方便消融实验.
    Args:
        seed :  int
    '''
    # 固定 scipy 的随机种子
    random.seed(seed)  # 固定 random 库的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 python hash 的随机性（并不一定有效）
    np.random.seed(seed)  # 固定 numpy  的随机种子
    torch.manual_seed(seed)  # 固定 torch cpu 计算的随机种子
    torch.cuda.manual_seed(seed)  # 固定 cuda 计算的随机种子
    torch.backends.cudnn.deterministic = True  # 是否将卷积算子的计算实现固定。torch 的底层有不同的库来实现卷积算子
    torch.backends.cudnn.benchmark = True  # 是否开启自动优化，选择最快的卷积计算方法

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def checkpoint(model, epoch, model_path, logger, method_name, fold):
    model_out_path = model_path + "/model_epoch_{}_fold_{}_{}.pth".format(method_name, fold, epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False, accum_iter=2):
    '''训练集每个epoch训练函数
    Args:
        epoch : int , 训练到第几个 epoch
        model : object, 需要训练的模型
        loss_fn : object, 损失函数
        optimizer : object, 优化方法
        train_loader : object, 训练集数据生成器
        scaler : object, 梯度放大器
        device : str , 使用的训练设备 e.g 'cuda:0'
        scheduler : object , 学习率调整策略
        schd_batch_update : bool, 如果是 true 则每一个 batch 都调整，否则等一个 epoch 结束后再调整
        accum_iter : int , 梯度累加
    '''

    model.train()  # 开启训练模式

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # 构造进度条
    epoch_accuracy = 0.0
    for step, (imgs, image_labels) in pbar:  # 遍历每个 batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():  # 开启自动混精度
            image_preds = model(imgs)  # 前向传播，计算预测值
            loss = loss_fn(image_preds, image_labels)  # 计算 loss

        scaler.scale(loss).backward()  # 对 loss scale, scale梯度
        accuracy = (image_preds.argmax(dim=1) == image_labels).float().mean()
        epoch_accuracy += accuracy
        # loss 正则,使用指数平均
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
            scaler.step(optimizer)  # unscale 梯度, 如果梯度没有 overflow, 使用 opt 更新梯度, 否则不更新
            scaler.update()  # 等着下次 scale 梯度
            optimizer.zero_grad()  # 梯度清空

            if scheduler is not None and schd_batch_update:  # 学习率调整策略
                scheduler.step()

        # 打印 loss 值
        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:  # 学习率调整策略
        scheduler.step()
    return running_loss, epoch_accuracy / len(train_loader)

def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    '''验证集 inference
    Args:
        epoch : int, 第几个 epoch
        model : object, 模型
        loss_fn : object, 损失函数
        val_loader ： object, 验证集数据生成器
        device : str , 使用的训练设备 e.g 'cuda:0'
    '''

    model.eval()  # 开启推断模式

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # 构造进度条

    for step, (imgs, image_labels) in pbar:  # 遍历每个 batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # 前向传播，计算预测值
        image_preds_all += [
            torch.argmax(image_preds, 1).detach().cpu().numpy()
        ]  # 获取预测标签
        image_targets_all += [image_labels.detach().cpu().numpy()]  # 获取真实标签

        loss = loss_fn(image_preds, image_labels)  # 计算损失

        loss_sum += loss.item() * image_labels.shape[0]  # 计算损失和
        sample_num += image_labels.shape[0]  # 样本数

        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'  # 打印平均 loss
        pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format( (image_preds_all == image_targets_all).mean()))  # 打印准确率
    return loss_sum/sample_num, image_targets_all, image_preds_all, (image_preds_all == image_targets_all).mean()


if __name__ == '__main__':
    pass
