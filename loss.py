import torch
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import torch.nn as nn
import cv2

def psnr(img, imclean):
    # Ensure values are between 0 and 1
    img = torch.clamp(img, 0, 1)
    imclean = torch.clamp(imclean, 0, 1)

    # Convert to numpy arrays
    Img = img.cpu().numpy().astype(np.float32)
    Iclean = imclean.cpu().numpy().astype(np.float32)

    # Compute PSNR
    PSNR = []
    for i in range(Img.shape[0]):
        ps = compare_psnr(Iclean[i], Img[i], data_range=1.0)
        if not np.isinf(ps):
            PSNR.append(ps)
    return np.mean(PSNR)

def ssim(img, imclean):
    # Ensure values are between 0 and 1
    img = torch.clamp(img, 0, 1)
    imclean = torch.clamp(imclean, 0, 1)

    # Convert to numpy arrays
    Img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    # 
    Iclean = np.squeeze(Iclean)
    Img = np.squeeze(Img)

    # Compute SSIM
    SSIM = np.mean([compare_ssim(Iclean[i], Img[i],data_range=1.0) for i in range(Img.shape[0])])

    return SSIM


mse = nn.MSELoss(reduction='mean') 

def loss_aug(clean, clean1, noise_w, noise_w1, noise_b, noise_b1):
    loss1 = mse(clean1,clean)
    loss2 = mse(noise_w1,noise_w)
    loss3 = mse(noise_b1,noise_b)
    loss = loss1 + loss2 + loss3
    return loss

# get sift
def test(src, model,opt=0):
    
    img1 = (src*255.0).astype('uint8')
    img2 = (model*255.0).astype('uint8')
    sift = cv2.SIFT_create()  
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  
    try:
        matches = flann.knnMatch(des1, des2, k=2)  
    except:
        return 0
    good_matches = []
    temp = []
    for i,(m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:  
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            if abs(pt1[1]-pt2[1]) < 15:
                temp.append([int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])])
                good_matches.append(m)
    num=len(good_matches)
    if opt == 0:
        if num == 0:
            return 0
        else:
            return temp
    else:
        return 1

def generate_sift(x_train,y_train):
    n = np.zeros((x_train.shape[0],176,36))
    for i in range(x_train.shape[0]):
        x = test(x_train[i,0,:,:],y_train[i,0,:,:],0)
        if x == 0:
            continue
        for k in x:
            n[i,k[1],k[0]] = 1
    n = n.reshape([n.shape[0],1,176,36])
    return n

def match_loss(clean,input_clear,input_noisy):
    clean_output = clean.cpu().detach().numpy()
    clean_input = input_clear.cpu().detach().numpy()
    input_noisy = input_noisy.cpu().detach().numpy()
    
    temp = 0.0
    p = []
    n = test(clean_output[0,:,:],clean_input[0,:,:],0)
    n1 = test(input_noisy[0,:,:],clean_input[0,:,:],0)
    
    # print(n)
    # print(n1)
    
    r = n - n1
    if r  > 0:
        temp += 0.0
    elif r == 0:
        temp += 1.0
    elif r < 0:
        temp += 2.0
    
    return temp*0.003
    
def loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2):
    loss1 = mse(input_noisy_pred, input_noisy)
    
    loss2 = mse(clean1,clean)
    loss3 = mse(noise_b3, noise_b)
    loss4 = mse(noise_w2, noise_w)
    loss5 = mse(clean2, clean)
    
    loss6 = mse(clean3, torch.zeros_like(clean3))
    loss7 = mse(noise_w1, torch.zeros_like(noise_w1))
    loss8 = mse(noise_b1, torch.zeros_like(noise_b1))
    loss9 = mse(noise_b2, torch.zeros_like(noise_b2))

    loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
    return loss


def mes_loss(clean,input_clear,level):
    loss = 0
    
    for i in range(level.shape[0]):
        loss_rec = mse(clean[i],input_clear[i])
        if level[i] == 0:
            loss += loss_rec * 1.5
        elif level[i] == 1:
            loss += loss_rec * 1.25
        elif level[i] == 2:
            loss += loss_rec * 0.9
        elif level[i] == 3:
            loss += loss_rec * 0.7
        elif level[i] == 4:
            loss += loss_rec * 0.5
        
        
    
    return loss / level.shape[0]

