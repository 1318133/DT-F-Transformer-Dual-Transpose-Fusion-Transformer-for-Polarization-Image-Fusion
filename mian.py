import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim, sigmoid, zeros_like
from tqdm import tqdm
from imageio import imwrite
from eval import eval_net

import torch.nn.functional as F

from unet import palfu

import pytorch_ssim
from pytorch_msssim import ssim
import cv2

# from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, TestDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/home/s1u1/dataset/Pal_datasets/KAUST_Pol_dataset/DoLP_g_128/'
dir_mask = '/home/s1u1/dataset/Pal_datasets/KAUST_Pol_dataset/S_g_128/'


dir_checkpoint = 'checkpoints/'
showpathimg = 'epoch_fuseimg_show/img/'
showpathfxo = 'epoch_fuseimg_show/fxo/'
showpathfyo = 'epoch_fuseimg_show/fyo/'

SSIM_WEIGHTS = [1, 10, 100, 1000]

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
loss_mse = nn.MSELoss(reduction='mean').cuda()

def features_grad(features):
    kernel = [[1, 1, 1 ], [1, -1, 1], [1, 1 , 1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    feat_grads = torch.abs(feat_grads)
    return feat_grads


def imgshow(img, showpath, index):
    img = img[1,:,:,:]
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1,2,0)
    img = img.astype('uint8')
    if img.shape[2] == 1:
        img = img.reshape([img.shape[0], img.shape[1]])
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    return img

def base( img, imgs):#S0,DoPL
    imgs = sig_norm(imgs)
    loss_1 = (1 - ssim(img, imgs))
    loss_1 = torch.mean(loss_1)
    # g_m = torch.sigmoid(features_grad(img))
    # g_i = torch.sigmoid(features_grad(imgs))
    # g_t =  torch.sigmoid(features_grad(true_masks))
    # g = torch.where(g_i>g_t,g_i,g_t)
    # loss_1 = mae_loss(g_m,g)

    loss_2 = loss_mse(img, imgs) 
    loss_2 = torch.mean(loss_2)

    loss = loss_1+10*loss_2#loss_1 + 10 * loss_2
    return loss

def loss_set( img, imgs, true_masks):#S0,DoPL

    g_i = features_grad(imgs)
    g_m = features_grad(true_masks)
    g_img = features_grad(img)

    zero = torch.zeros_like(img)
    one = torch.ones_like(img)

    map1 = torch.where(g_i>g_m,one,zero)
    map2 = 1-map1

    # map1 = torch.sigmoid(map1-map1.mean())#
    # map1 = (map1-map1.min())/(map1.max()-map1.min())
    # map2 = torch.sigmoid(map2-map2.mean())#
    # map2 = (map2-map2.min())/(map2.max()-map2.min())

    imgs = normal(torch.sigmoid(imgs-imgs.mean()))
    true_masks = normal(torch.sigmoid(true_masks-true_masks.mean()))
    
    lossi = mae_loss(map1*img,map1*imgs)
    lossm = mae_loss(map2*img,map2*true_masks)
    loss = lossi+lossm
    # loss = loss1+loss2+loss3
    return loss

def normal(x):
    x = (x-x.min())/(x.max()-x.min())
    return x

def sig_norm(map1):
    map1 = torch.sigmoid((map1-map1.mean())*5)#
    map1 = (map1-map1.min())/(map1.max()-map1.min())
    return map1

def loss_grad( img, imgs, true_masks):
    g_i = normal(features_grad(imgs))
    g_m = normal(features_grad(true_masks))
    g_img = normal(features_grad(img))
    i = 2*g_i-g_i*g_i
    m = 2*g_m-g_m*g_m
    g_max = torch.where(i>m,i,m)
    loss = mae_loss(g_img,g_max)
    return loss


def train_net(net, 
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.25):
    ph = 1
    al = 0 #低于此全部置0
    c = 3500

    dataset = BasicDataset(dir_img, dir_mask, img_scale,type='png')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train, val1 = random_split(dataset, [n_train, n_val])
    n_val9 = int(n_val*0.5)
    n_val1 = n_val-n_val9
    val2, val = random_split(val1, [n_val9, n_val1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    index = 1
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    # sloss = sobelloss()
    # sloss2 = sobelloss2()
    # usloss = unsameloss()
    # up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    # down2 = nn.MaxPool2d(2)
    # down4 = nn.MaxPool2d(4)
    # down8 = nn.MaxPool2d(8)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-7, momentum=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-7)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    for epoch in range(epochs):
        net.train()

        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs2 = batch['image2']
                true_masks2 = batch['mask2']
                imgs3 = batch['image3']
                true_masks3 = batch['mask3']
                imgs4 = batch['image4']
                true_masks4 = batch['mask4']
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    imgs2 = imgs2.cuda()
                    true_masks2 = true_masks2.cuda()
                    imgs3 = imgs3.cuda()
                    true_masks3 = true_masks3.cuda()
                    imgs4 = imgs4.cuda()
                    true_masks4 = true_masks4.cuda()
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                imgs2 = imgs2.to(device=device, dtype=torch.float32)
                true_masks2 = true_masks2.to(device=device, dtype=torch.float32)
                imgs3 = imgs3.to(device=device, dtype=torch.float32)
                true_masks3 = true_masks3.to(device=device, dtype=torch.float32)
                imgs4 = imgs4.to(device=device, dtype=torch.float32)
                true_masks4 = true_masks4.to(device=device, dtype=torch.float32)

                # img,img_y = net(true_masks, imgs)#(imgs, true_masks)#
                img,x_mi,_ = net(imgs, true_masks)#(true_masks, imgs)# SAR OPT

############Pal fu loss 
                # loss_s = loss_set(img,imgs,true_masks)
                loss_g = loss_grad(img,imgs,true_masks)
                # loss_d = loss_degree(img)
                # loss_m = loss_mean(img)
                loss1 = base(img,imgs)
                loss2 = base(img, true_masks)
                # loss_c,loss1,loss2 = loss_caculate( img, true_masks, imgs)
                lossall = loss1+loss2+loss_g#+loss_c#10*loss_s#+loss_d+loss_m

                pbar.set_postfix(**{'loss1 (batch)': loss1.item(),'loss2 (batch)': loss2.item(),'loss3 (batch)': loss_g.item()})
                optimizer.zero_grad()
                lossall.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                ######
                if global_step == ((n_train // batch_size)*index):
                    g = imgshow(img,showpathimg,index)

                    g3 = imgshow(imgs,showpathfxo,index)
                    g4 = imgshow(true_masks,showpathfyo,index)

                    print(optimizer.state_dict()['param_groups'][0]['lr'])
#################
                    index += 1  
                #####
                if global_step % (n_train // (1  * batch_size)) == 0:

                    scheduler.step()


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batchsize', metavar='B', type=int, nargs='?', default=40,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=str, default=True,
                        help='If test images turn True, train images turn False')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    pthf = None#'/home/s1u1/code/pal-chaunshu/checkpoints/CP_epoch60.pth'

    vi = '/home/s1u1/dataset/Pal_datasets/KAUST_Pol_dataset/DoLP_g_s/'
    ir = '/home/s1u1/dataset/Pal_datasets/KAUST_Pol_dataset/S_g_s/'

    path = '/home/s1u1/code/pal-chaunshu/result/'

    pathadd = './outputsadd/'


    dataset = TestDataset(ir, vi)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    net = palfu.wtNet(n_channels=1, n_classes=1, bilinear=True, pthfile=pthf)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    index = 1
    Time = 0
    if args.test:
        for im in test_loader:
            
            ir = im['image']
            vi = im['mask']
            if torch.cuda.is_available():
                ir = ir.cuda()
                vi = vi.cuda()
            # Net = Wdenet.wtNet(1, 1)
            # Net = unet_model.UNet(1,1)
            Net = palfu.wtNet(1, 1, pthfile=pthf)
            Net = Net.cuda()
            ##########################
            add = ir*0.5 + vi*0.5
            img_final = add.detach().cpu().numpy()
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(1, 2, 3, 0)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png' 
            path_out = pathadd + file_name
            # index += 1            
            imwrite(path_out, img)
            ##########################
            start = time.time()
            # img, _, _ = Net(vi, ir)
            img,_,_ = Net(vi, ir)
            zero = torch.zeros_like(img)
            img = torch.where(img<0,zero,img)
            # img = Net(vi, ir)
            img_final = img.detach().cpu().numpy() 
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(1, 2, 3, 0)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            end = time.time()
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = path + file_name
            # index += 1            
            imwrite(path_out, img)
            Time += end-start
            # print(index)
            # print(end-start)
            index += 1  
        average_time = Time/(len(test_loader))  
        print(average_time) 

    else:
        try:
            train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
