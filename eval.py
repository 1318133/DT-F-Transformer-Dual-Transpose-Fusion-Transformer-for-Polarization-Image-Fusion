import torch
import torch.nn.functional as F
import pytorch_ssim
import torch.nn as nn
from pytorch_msssim import ssim
from tqdm import tqdm
# from train import caculateloss_v2
from dice_loss import dice_coeff

SSIM_WEIGHTS = [1, 10, 100, 1000]

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
loss_mse = nn.MSELoss(reduction='mean').cuda()

def midfilt(features):
    kernel = torch.ones(3, 3)
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
    return feat_grads

def fusion_loss(img_1,img_2):
    # img1 = torch.from_numpy(np.rollaxis(img_1, 2)).float().unsqueeze(0)
    # img2 = torch.from_numpy(np.rollaxis(img_2, 2)).float().unsqueeze(0)

    if torch.cuda.is_available():
        img1 = img_1.cuda()
        img2 = img_2.cuda()
    # print(pytorch_ssim.ssim(img1, img2))
    ssimloss = pytorch_ssim.SSIM(window_size = 11)

    ssim_weight = SSIM_WEIGHTS[2]
    ssim_loss_value = ssimloss(img1, img2)
    pixel_loss = mae_loss(img1,img2)
    ssim_loss = 1 - ssim_loss_value
    stay_loss = ssim_loss + 20*pixel_loss

    return stay_loss

def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
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
    return feat_grads

def caculateloss(fxout, fyout, img, imgs, true_masks):
    m1 = torch.mean(features_grad(fxout), dim=[1])
    weight1 = torch.unsqueeze(m1, dim=1)
    weight1 = torch.norm(weight1)
                # weight1 = torch.unsqueeze(weight1, dim=1)
                # m2 = torch.mean(fyout.pow(2), dim=[1])
    m2 = torch.mean(features_grad(fyout), dim=[1])
    weight2 = torch.unsqueeze(m2, dim=1)
    weight2 = torch.norm(weight2)
                # weight1 = torch.unsqueeze(weight1, dim=1)
    weight1 = weight1/(weight1+weight2)
    weight2 = weight2/(weight1+weight2)
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    loss = loss_1 + 20 * loss_2
    return loss

def caculateloss_v2(fxout, fyout, img, imgs, true_masks):
    x1 = torch.mean(torch.mean(fxout,dim=2),dim=2)
    x2 = torch.mean(torch.mean(fyout,dim=2),dim=2)
    a = x1.shape[0]
    x1 = torch.reshape(x1,[a,1,1,1])
    x2 = torch.reshape(x2,[a,1,1,1])
    bb1 = torch.abs(torch.mean(torch.tanh(fxout-x1)))
    bb2 = torch.abs(torch.mean(torch.tanh(fyout-x2)))
    b1 = bb1/(bb1+bb2)
    b2 = bb2/(bb1+bb2)
    bw1 = img*b1
    bimgs = imgs*b1
    bw2 = img*b2
    btrue_masks = true_masks*b2
    loss_3 = loss_mse(bw1, bimgs) + loss_mse(bw2, btrue_masks)
    loss_3 = torch.mean(loss_3)

    m1 = torch.mean(features_grad(torch.tanh(fxout)), dim=[1])
    weight1 = torch.unsqueeze(m1, dim=1)
    weight1 = torch.sigmoid(weight1)#.pow(2)
                # weight1 = torch.unsqueeze(weight1, dim=1)
                # m2 = torch.mean(fyout.pow(2), dim=[1])
    m2 = torch.mean(features_grad(torch.tanh(fyout)), dim=[1])
    weight2 = torch.unsqueeze(m2, dim=1)
    weight2 = torch.sigmoid(weight2)#.pow(2)
                # weight1 = torch.unsqueeze(weight1, dim=1)
    weight1 = weight1/(weight1+weight2)
    weight2 = weight2/(weight1+weight2)
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    loss = loss_1 + 20 * loss_2 + 0.2 * loss_3
    return loss

def caculateloss_v3(fxout, fyout, img, imgs, true_masks):
    m1 = features_grad(fxout)
    m1abs = torch.abs(m1)
    m2 = features_grad(fyout)
    m2abs = torch.abs(m2)
    mone = torch.ones_like(m1)
    mzero = torch.zeros_like(m1)
    mhaf = torch.zeros_like(m1)*0.5
    mask1 = torch.where(m1abs>m2abs,mone,mzero)
    mask1 = torch.where(m1abs==m2abs,mhaf,mask1)
    mask2 = 1-mask1
    weight1 = mask1#torch.sigmoid((m1abs-m2abs)*mask1*10000)+(1-torch.sigmoid((m2abs-m1abs)*mask2*10000))
    # weight1 = midfilt(weight1)
    # weight2 = mask2#torch.abs(torch.sigmoid((m1abs-m2abs)*mask1))+(1-torch.abs(torch.sigmoid((m2abs-m1abs)*mask2)))
    weight2 = mask2#1-weight1
    # torch.sigmoid(m1-torch.mean(m1))-torch.sigmoid(m2-torch.mean(m2))
    # weight1 = torch.sigmoid((torch.sqrt(torch.abs(m1))-torch.sqrt(torch.abs(m2)))*10)
    # bach, w, h = weight1.shape[0],weight1.shape[1],weight1.shape[2]
    # weight1 = torch.reshape(weight1, [bach,1,w,h])
    # weight2 = 1-weight1
    
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    # bb1 = torch.sigmoid(fxout)
    # bb2 = torch.sigmoid(fyout)
    x1 = torch.mean(fxout)
    x2 = torch.mean(fyout)
    # x1 = torch.mean(torch.mean(fxout,dim=2),dim=2)
    # x2 = torch.mean(torch.mean(fyout,dim=2),dim=2)
    # a = x1.shape[0]
    # x1 = torch.reshape(x1,[a,1,1,1])
    # x2 = torch.reshape(x2,[a,1,1,1])
    bb1 = torch.sqrt(torch.sigmoid(fxout-x1))
    bb2 = torch.sqrt(torch.sigmoid(fyout-x2))
    # bb1 = torch.abs(torch.mean(torch.tanh(fxout-x1)))
    # bb2 = torch.abs(torch.mean(torch.tanh(fyout-x2)))
    b1 = bb1/(bb1+bb2)
    b2 = bb2/(bb1+bb2)
    b1 = torch.sigmoid(b1-torch.mean(b1))
    b2 = torch.sigmoid(b2-torch.mean(b2))
    # b1 = 0.5
    # b2 = 0.5
    bw1 = img*b1
    bimgs = imgs*b1
    bw2 = img*b2
    btrue_masks = true_masks*b2
    loss_3 = loss_mse(bw1, bimgs) + loss_mse(bw2, btrue_masks)
    loss_3 = torch.mean(loss_3)

    loss = loss_1 + 20 * loss_2 + 0.4*loss_3
    return loss

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot1 = 0
    tot2 = 0
    tot4 = 0
    tot8 = 0
    down2 = nn.MaxPool2d(2)
    down4 = nn.MaxPool2d(4)
    down8 = nn.MaxPool2d(8)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, imgsr = batch['image'], batch['mask']
            imgs2, imgsr2 = batch['image2'], batch['mask2']
            imgs3, imgsr3 = batch['image3'], batch['mask3']
            imgs4, imgsr4 = batch['image4'], batch['mask4']
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgsr = imgsr.to(device=device, dtype=mask_type)
            imgs2 = imgs2.to(device=device, dtype=torch.float32)
            imgsr2 = imgsr2.to(device=device, dtype=mask_type)
            imgs3 = imgs3.to(device=device, dtype=torch.float32)
            imgsr3 = imgsr3.to(device=device, dtype=mask_type)
            imgs4 = imgs4.to(device=device, dtype=torch.float32)
            imgsr4 = imgsr4.to(device=device, dtype=mask_type)

            with torch.no_grad():
                x,y,z,q,_,x1,y1 = net(imgs, imgsr)
                # _,vis,inf,_,_,_,_ = net(imgs, imgsr)
                # mask_pred = net(imgs)

            # tot1 += fusion_loss(imgs, vis).item()
            # tot2 += fusion_loss(imgsr, inf).item()
            # tot = max(tot1,tot2)
            x2 = down2(x1)
            y2 = down2(y1)
            x4 = down4(x1)
            y4 = down4(y1)
            x8 = down8(x1)
            y8 = down8(y1)
            tot1 += caculateloss_v3(x1, y1, x, imgs, imgsr).item()
            # tot2 += caculateloss(x2, y2, y, imgs2, imgsr2).item()
            # tot4 += caculateloss(x4, y4, z, imgs3, imgsr3).item()
            # tot8 += caculateloss(x8, y8, q, imgs4, imgsr4).item()
            # tot = max(tot1,tot2,tot4,tot8)

            # if net.n_classes > 1:
            #     tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
            #     pred = torch.sigmoid(mask_pred)
            #     pred = (pred > 0.5).float()
            #     tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot1 / n_val
