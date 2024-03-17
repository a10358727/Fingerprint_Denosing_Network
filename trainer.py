import torch
import os
import tqdm
import cv2
import numpy as np
import torch.nn as nn
from model import MLDN_model
from loss import loss_aug, loss_main,ssim,psnr,generate_sift,match_loss,mes_loss
from data import dataloader
from torch.backends import cudnn

mse = nn.MSELoss(reduction='mean') 





def save_image(opt,index,noise_output,clean_output,level):
    
    result_folder = './results/' + opt.name + '/test/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_folder += opt.model_name + '/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder + '0/'):
        os.makedirs(result_folder + '0/')
    if not os.path.exists(result_folder + '1/'):
        os.makedirs(result_folder + '1/')
    if not os.path.exists(result_folder + '2/'):
        os.makedirs(result_folder + '2/')
    if not os.path.exists(result_folder + '3/'):
        os.makedirs(result_folder + '3/')
    if not os.path.exists(result_folder + '4/'):
        os.makedirs(result_folder + '4/')
    img = _save_image(opt,noise_output)
    cv2.imwrite(result_folder + str(level.item()) + '/noise_' + str(index) + '.bmp', img.astype(np.uint8))
    img = _save_image(opt,clean_output)
    cv2.imwrite(result_folder + str(level.item()) + '/clean_' + str(index) + '.bmp', img.astype(np.uint8))

def _save_image(opt,img):
    img = img.cpu().detach().numpy()
    img = img[:,:]
    img = img.reshape(opt.output_h,opt.output_w)
    img = img  * 255.0
    img = img.astype('uint8')
    return img 
    

def train(opt,model,device,train_dataloader,val_dataloader):
    epoch_num = opt.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-09, amsgrad=True)
    for epoch in (range(1,epoch_num+1)):
        
        model.train()
        loss_train = 0.0
        
        for index, data in enumerate(tqdm.tqdm(train_dataloader)):
            
            
            noise_data,clear_data,level = data
            noise_data = noise_data.numpy()
            clear_data = clear_data.numpy()
            level = level.numpy()
            
            # print(noise_data.shape)
            # print(clear_data.shape)
            # print(level.shape)
            
            level = torch.from_numpy(level)
            level = level.to(device)

            sift = generate_sift(noise_data,clear_data)
            sift = torch.from_numpy(sift)
            sift = sift.float().to(device)
            
            input_noisy = torch.from_numpy(noise_data)       # torch.Tensor
            input_clear = torch.from_numpy(clear_data)
            input_noisy = input_noisy.float()
            input_noisy = input_noisy.to(device)
            input_clear = input_clear.float()
            input_clear = input_clear.to(device)
            
            optimizer.zero_grad()
            

            noise_w, noise_b, clean = model(input_noisy,sift,level)
            noise_w1, noise_b1, clean1 = model((clean),sift,level)
            noise_w2, noise_b2, clean2 = model((clean + noise_w),sift,level) 
            noise_w3, noise_b3, clean3 = model((noise_b),sift,level)

            #stage3
            #noise_w4, noise_b4, clean4 = model((clean+torch.pow(clean,gamma)*noise_w-noise_b),sift,level) #2
            #noise_w5, noise_b5, clean5 = model((clean-torch.pow(clean,gamma)*noise_w+noise_b),sift,level) #3
            #noise_w6, noise_b6, clean6 = model((clean-torch.pow(clean,gamma)*noise_w-noise_b),sift,level) #4
            #noise_w7, noise_b7, clean7 = model((clean+noise_w+noise_b),sift,level) #5

            input_noisy_pred = clean+noise_w+noise_b
    
            loss_match  = match_loss(clean,input_clear,input_noisy)
            loss_rec = mes_loss(clean,input_clear,level)
            loss = loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2)

            #stage3
            #loss_aug1 = loss_aug(clean, clean4, noise_w, noise_w4, noise_b, -noise_b4)
            #loss_aug2 = loss_aug(clean, clean5, noise_w, -noise_w5, noise_b, noise_b5)
            #loss_aug3 = loss_aug(clean, clean6, noise_w, -noise_w6, noise_b, -noise_b6)
            #loss_aug4 = loss_aug(clean, clean7, noise_w, noise_w7, noise_b, noise_b7)
            
            # if level == 1:
            #     loss_total = loss + loss_rec * 0.5 + loss_match
            # else:
            #     loss_total = loss + loss_rec * 1.25 + loss_match
            #stage3
            #loss_total = loss + 0.1(loss_aug1+loss_aug2+loss_aug3+loss_aug4)+ loss_rec * 0.5 + loss_match
            loss_total = loss + loss_rec  + loss_match            
            loss_total.backward()
            optimizer.step()
            loss_train += loss_total.item()

        print('Trainging Loss', loss_train/len(train_dataloader))
        
        if epoch % opt.save_epoch_freq == 0:
            torch.save(model.state_dict(), './results/'+ opt.name+ '/weights/'+str(epoch)+'.pt')

        # Validation
        model.eval()
        val_loss = 0.0
        ssim_total = 0.0
        psnr_total = 0.0
        with torch.no_grad():
            for index,data in enumerate(tqdm.tqdm(val_dataloader)):
                noise_val, clear_val, level = data
                
                noise_val = noise_val.numpy()
                clear_val = clear_val.numpy()
                level = level.numpy()
                
                sift = generate_sift(noise_val,clear_val)
                sift = torch.from_numpy(sift)
                sift = sift.float().to(device)
            
                noise_val = torch.from_numpy(noise_val)       
                clear_val = torch.from_numpy(clear_val)
                level = torch.from_numpy(level)
                
                level = level.to(device)
                noise_val = noise_val.float().to(device)
                clear_val = clear_val.float().to(device)
                
                noise_w, noise_b, clean = model(noise_val,sift,level)
                noise_w1, noise_b1, clean1 = model((clean),sift,level)
                noise_w2, noise_b2, clean2 = model((clean + noise_w),sift,level) 
                noise_w3, noise_b3, clean3 = model((noise_b),sift,level)
                
                input_noisy_pred = clean+noise_w+noise_b
                loss_match  = match_loss(clean,clear_val,noise_val)
                loss_rec = mes_loss(clean,clear_val,level)
                loss = loss_main(noise_val, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2)

                loss_clear_val = loss + loss_rec + loss_match
                
                ssim_total += ssim(clean, noise_val)
                psnr_total += psnr(clean, noise_val)
                
                val_loss += loss_clear_val.item()

        print('Epoch: {}, Validation Loss : {}'.format(epoch, val_loss / len(val_dataloader)))
        print('Epoch: {}, SSIM Value : {}'.format(epoch, ssim_total / len(val_dataloader)))
        print('Epoch: {}, PSNR Value : {}'.format(epoch, psnr_total / len(val_dataloader)))
        
def test(opt,model,device,test_dataloader):
    
    model.eval()
    val_loss = 0.0
    ssim_total = 0.0
    psnr_total = 0.0
    with torch.no_grad():
        for index,val_data in enumerate(tqdm.tqdm(test_dataloader)):
            noise_val, clear_val ,level = val_data

            noise_val = noise_val.numpy()
            clear_val = clear_val.numpy()
            level = level.numpy()
                
            sift = generate_sift(noise_val,clear_val)
            sift = torch.from_numpy(sift)
            sift = sift.float().to(device)
            
            noise_val = torch.from_numpy(noise_val)       
            clear_val = torch.from_numpy(clear_val)
            level = torch.from_numpy(level)
                
            level = level.to(device)
            noise_val = noise_val.float().to(device)
            clear_val = clear_val.float().to(device)
            
            
            noise_w_val, noise_b_val, clean_val = model(noise_val,sift,level)
            # clean_val = noise_val
            
            save_image(opt,index,noise_val,clean_val,level)

            loss_clear_val = mse(clean_val, noise_val)

            ssim_total += ssim(clean_val, noise_val)
            psnr_total += psnr(clean_val, noise_val)
            val_loss += loss_clear_val.item()

        print(' Test Loss: {}'.format( val_loss / len(test_dataloader)))
        print(' SSIM Value : {}'.format( ssim_total / len(test_dataloader)))
        print(' PSNR Value : {}'.format( psnr_total / len(test_dataloader)))
        

def trainer(opt):
    # save result path
    result_path = './results/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_path + opt.name + '/'):
        os.makedirs(result_path + opt.name + '/')
    if not os.path.exists(result_path + opt.name + '/weights/'):
        os.makedirs(result_path + opt.name + '/weights/')

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create cvf model
    model = MLDN_model()
    if opt.Continue:
        model.load_state_dict(torch.load('./results/' + opt.name + '/weights/' + opt.model_name + '.pt'))
    
    model = model.to(device)
    # create train and  val dataloader
    train_dataloader,val_dataloader = dataloader.train_dataloader(opt)
    # start train 
    print('start train')

    train(opt,model,device,train_dataloader,val_dataloader)

def tester(opt):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create cvf model
    model = MLDN_model()
    # load model 
    model.load_state_dict(torch.load('./results/' + opt.name + '/weights/' + opt.model_name + '.pt'))
    
    model = model.to(device)
    # create test dataloader
    test_dataloader = dataloader.test_dataloader(opt)

    test(opt,model,device,test_dataloader)
    
