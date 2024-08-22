import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
from torch import nn
import datetime
import pandas as pd
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import timm
from sklearn import model_selection
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from architecture import *
from utils import *
from losses import *
from opts.data_loader import *
from opts.fmix import *



def main():
    rand_seed = 666
    seed_everything(rand_seed)
    device = torch.device(opt.device)

    # dataset
    train = pd.read_csv(opt.train_csv_path)
    # train_df, valid_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.label.values)
    folds = StratifiedKFold(n_splits=opt.fold_num, shuffle=True, random_state=rand_seed).split(
        np.arange(train.shape[0]), train.label.values)
    trn_transform = get_train_transforms(img_size=opt.img_size)
    val_transform = get_valid_transforms(img_size=opt.img_size)

    # saving path
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = opt.outf + date_time + '/results/'
    model_path = opt.outf + date_time + '/models/'
    log_path = opt.outf + date_time + '/logs/'
    writer_train = SummaryWriter(log_path+ '/train')
    writer_val = SummaryWriter(log_path+ '/val')
    writer_train_acc = SummaryWriter(log_path+ '/train_acc')
    writer_val_acc = SummaryWriter(log_path + '/val_acc')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # log
    logger = gen_log(log_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))

    accuracy_max = 0.0
    fold_num = 0
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold == fold_num:
            logger.info('Training with {} started'.format(fold))
            logger.info('Train : {}, Val : {}'.format(len(trn_idx), len(val_idx)))
            train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root=opt.train_img_path,
                                                          trn_transform=trn_transform,
                                                          val_transform=val_transform, bs=opt.batch_size, n_job=0)
            # model
            model = model_generator(opt.method, opt.pretrained_model_path, pretrained=opt.pretrained).to(device)
            scaler = GradScaler()
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

            if opt.scheduler == 'MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
            elif opt.scheduler == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
            elif opt.scheduler == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1,
                                                                                 eta_min=1e-6, last_epoch=-1)

            loss_tr = nn.CrossEntropyLoss().to(device)
            loss_fn = nn.CrossEntropyLoss().to(device)

            for epoch in tqdm(range(opt.max_epoch)):
                loss_train, acc_train = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scaler, scheduler=scheduler, schd_batch_update=False)
                with torch.no_grad():
                    loss_val, targets, preds, accuracy = valid_one_epoch(epoch, model, loss_fn, val_loader, device)
                logger.info("===> Epoch {} Complete: Train_Loss: {:.6f} Val_Loss: {:.6f} lr: {:.6f} Train_Accuracy: {:.4f}  Valid_Accuracy: {:.4f}".
                    format(epoch, loss_train, loss_val, optimizer.param_groups[0]["lr"], acc_train, accuracy))

                writer_train.add_scalar('loss', loss_train, epoch)
                writer_val.add_scalar('loss', loss_val, epoch)

                writer_train_acc.add_scalar('accuracy', acc_train, epoch)
                writer_val_acc.add_scalar('accuracy', accuracy, epoch)
                if accuracy > accuracy_max:
                    logger.info('--------------------------------------------')
                    logger.info('--------------------------------------------')
                    logger.info('     Val:       ')
                    logger.info(
                        "===> Epoch {} Complete: Train_Loss: {:.6f} Val_Loss: {:.6f} lr: {:.6f} Train_Accuracy: {:.4f} Val_Accuracy: {:.4f}".
                            format(epoch, loss_train, loss_val, optimizer.param_groups[0]["lr"], acc_train,
                                   accuracy))
                    logger.info('--------------------------------------------')
                    logger.info('--------------------------------------------')

                    accuracy_max = accuracy
                    name = result_path + '/' + 'Test_{}_fold{}_{:.4f}'.format(epoch, fold, accuracy) + '.csv'
                    df = pd.DataFrame({'Label': targets, 'Pred': preds, 'Train_Accuracy': acc_train, 'Valid_Accuracy': accuracy })
                    df.to_csv(name)
                    checkpoint(model, epoch, model_path, logger, opt.method, fold)

            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()
    writer_train.close()
    writer_val.close()
    writer_train_acc.close()
    writer_val_acc.close()

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


