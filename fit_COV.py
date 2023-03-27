import os
import random
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tools_mine.Miou_COV import calculate_miou


def write_options(model_savedir, args, best_acc_val):
    aaa = []
    aaa.append(['lr', str(args.lr)])
    aaa.append(['batch', args.batch_size])
    aaa.append(['seed', args.batch_size])
    aaa.append(['best_val_acc', str(best_acc_val)])
    aaa.append(['warm_epoch', args.warm_epoch])
    aaa.append(['end_epoch', args.end_epoch])
    f = open(model_savedir + 'option' + '.txt', "a")
    for option_things in aaa:
        f.write(str(option_things) + '\n')
    f.close()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def fit(epoch, epochs, model, trainloader, valloader, device, criterion, optimizer, CosineLR):
    scaler = GradScaler()
    if torch.cuda.is_available():
        model.to('cuda')
    running_loss = 0
    model.train()
    train_pa_whole = 0
    train_iou_whole = 0
    for batch_idx, (imgs, masks) in enumerate(trainloader):
        imgs, masks_cuda = imgs.to(device), masks.to(device)
        imgs = imgs.float()
        with autocast():
            masks_pred = model(imgs)
            loss = criterion(masks_pred, masks_cuda)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        predicted = masks_pred.argmax(1)
        train_iou = calculate_miou(predicted, masks_cuda, 2)
        train_iou_whole += train_iou.item()
        running_loss += loss.item()
        epoch_iou = train_iou_whole / (batch_idx + 1)
    epoch_loss = running_loss / len(trainloader.dataset)
    val_running_loss = 0
    val_pa_whole = 0
    val_iou_whole = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(valloader):
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            predicted = masks_pred.argmax(1)
            val_iou = calculate_miou(predicted, masks_cuda, 2)
            val_iou_whole += val_iou.item()
            loss = criterion(masks_pred, masks_cuda)
            val_running_loss += loss.item()
            epoch_val_acc = val_pa_whole / (batch_idx + 1)
            epoch_val_iou = val_iou_whole / (batch_idx + 1)
    epoch_val_loss = val_running_loss / len(valloader.dataset)
    CosineLR.step()
    return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou
    
