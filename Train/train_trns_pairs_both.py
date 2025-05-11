import os
import FYP_package.transmorph2D.losses as losses
import FYP_package.transmorph2D.utils as utils

from FYP_package.voxelmorph.voxelmorph_timeseries.models.losses import BendingEnergy

from torch.utils.data import DataLoader


import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt


from FYP_package.transmorph2D.TransMorph import CONFIGS as CONFIGS_TM
import FYP_package.transmorph2D.TransMorph as TransMorph
from FYP_package.voxelmorph.voxelmorph_timeseries.dataloaders.heart_dataset import HeartDataset
from FYP_package.voxelmorph.voxelmorph_timeseries.dataloaders.npy_time_series_loader import TimeSeriesDataset

import wandb
import time
import argparse


def save_and_log_images(fixed, moving, flo, x_def, z_def, warped, epoch, batch_size, save_name="train_figure"):

    num_imgs = min(fixed.shape[0], 16) #better than using batch_size, as batch size may be smaller than 8 for last batch

    fixed_np   = fixed.cpu().detach().numpy()   # shape [B,H,W]
    moving_np  = moving.cpu().detach().numpy()
    flo_np   = flo.cpu().detach().numpy()
    x_def_np = x_def#.cpu().detach().numpy()
    z_def_np = z_def#.cpu().detach().numpy()
    warped_np  = warped.cpu().detach().numpy()

    fig, axes = plt.subplots(num_imgs, 7, figsize=(15, 3*num_imgs)) 
    for b in range(num_imgs):
        i = b
        j = 0
        img = fixed_np[i, 0]
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')     
        axes[i, j].set_title("Fixed")   
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = b
        j = 1
        img = moving_np[i, 0]
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')     
        axes[i, j].set_title("Moving")   
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 2
        img = flo_np[i, 0] 
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("Displacement X")    
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 3
        img = x_def[i][0] 
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("True disp X")    
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar


        i = i
        j = 4
        img = flo_np[i, 1]
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("Displacement Z")
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 5
        img = z_def[i][0] 
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("True disp Z")    
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 6
        img = warped_np[i, 0]    
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("Warped")    
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar
 
    plt.tight_layout()
    wandb.log({save_name: wandb.Image(fig), 'epoch': epoch})
 
    # filename = os.path.join(data_folder, f'epoch_{epoch}_images_grid.png')
    # plt.savefig(filename)
    plt.close(fig)

def main(args):

    if args.use_wandb == "True":
        wandb.init(project="transmorph_testing")


    weights = [1, args.gamma] # loss weights
    # save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # if not os.path.exists('experiments/'+save_dir):
    #     os.makedirs('experiments/'+save_dir)
    # if not os.path.exists('logs/'+save_dir):
    #     os.makedirs('logs/'+save_dir)

    lr = args.learning_rate # learning rate
    max_epoch = args.num_epochs #max traning epoch

    cont_training = False #if continue training
    epoch_start = 0

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph-Tiny']
    config.img_size = (args.image_height, args.image_width) # image size

    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        pass
        # epoch_start = 394
        # model_dir = 'experiments/'+save_dir
        # updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        # print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        # model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    data_file = os.path.expandvars(args.data_path)  #$TMPDIR D:/cardiac_dataset/reformattedreformatted

    if args.dataset == "synth":
        train_set = TimeSeriesDataset(data_file, n_frames=2, skip_frames=[args.frame_gap], get_def=True) #gets pairs of images with a gap of frame_gap
    elif args.dataset == "hearts":
        train_set = HeartDataset(data_file, frames=False, frame_gap=args.frame_gap)
    else:
        raise ValueError("Dataset not supported. Please use 'synth' or 'hearts'.")

    print(f'Training set size: {(len(train_set))}')    


    train_loader =  DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]

    if args.use_Bending_loss == "True":
        # add Bending energy loss
        bending_energy = BendingEnergy()
        criterions += [bending_energy.loss]
    else:
        # add Grad loss
        criterions += [losses.Grad(penalty='l2')]

    best_dsc = 0)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        model.train()
        loss_all = utils.AverageMeter()
        idx = 0
        t0 = time.perf_counter()
        for data, x_def, z_def in train_loader:
            t1 = time.perf_counter()
            idx += 1
            
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)]

            if args.dataset == "synth":
                x = data[:, 0, :, :].float().unsqueeze(1).cuda()
                y = data[:, 1, :, :].float().unsqueeze(1).cuda()

            elif args.dataset == "hearts":
                x = data[0].float().cuda()
                y = data[1].float().cuda()
            x_in = torch.cat((x,y), dim=1)
            t2 = time.perf_counter()
            output = model(x_in)
            t3 = time.perf_counter()
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(y, output[n]) * weights[n]  # swaped order to match bending loss
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            t4 = time.perf_counter()
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t5 = time.perf_counter()

            # This is done in the data set already, dont need to do this
            # del x_in
            # del output
            
            # # flip fixed and moving images
            # loss = 0
            # x_in = torch.cat((y, x), dim=1)
            # output = model(x_in)
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(x, output[n]) * weights[n]
            #     loss_vals[n] += curr_loss
            #     loss += curr_loss
            # loss_all.update(loss.item(), y.numel())
            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
            print(f'   Time to load input: {(t1-t0)*1000:.2f}ms, process input {(t2-t1)*1000:.2f}ms,   run model: {(t3-t2)*1000:.2f}ms,   compute loss: {(t4-t3)*1000:.2f}ms,   backprop: {(t5-t4)*1000:.2f}ms')
            if args.use_wandb == "True":

                if idx % 1 == 0:
                    wandb.log({
                                "train_loss": loss.item(),
                                "smooth_loss": loss_vals[1].item()/2,
                                "sim_loss": loss_vals[0].item()/2,
                                "epoch": epoch,
                            })
                
                if idx % args.log_freq == 0:
                    save_and_log_images(x, y, output[1], x_def, z_def, output[0], epoch, 8)

            t0 = time.perf_counter()


        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        if args.use_wandb == "True":
            wandb.log({
                            "Epoch_loss": loss_all.avg,
                        })
        
        #save the model and optimiser 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, f'{args.checkpoint_path}/epoch_{epoch}_checkpoint.pth.tar')


        loss_all.reset()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

# def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
#     torch.save(state, save_dir+filename)
#     model_lists = natsorted(glob.glob(save_dir + '*'))
#     while len(model_lists) > max_model_num:
#         os.remove(model_lists[0])
#         model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden-1)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden-1))
    print('If the GPU is available? ' + str(GPU_avai))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default= "D:/movies_npy/train", help="Path to the dataset")
    parser.add_argument("--dataset", type=str, default="synth", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.0001, help="Weighting of smoothness loss (0=no smoothing)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--image_height", type=int, default=320, help="Image height")
    parser.add_argument("--image_width", type=int, default=320, help="Image width")
    parser.add_argument("--checkpoint_path", type=str, default="chkpts5", help="Path to save the model checkpoint")
    parser.add_argument("--use_NCC_loss", type=str, default="True", help="Use NCC loss if true, otherwise use MSE")
    parser.add_argument("--use_Bending_loss", type=str, default="True", help="Use Bending energy if true, otherwise use Grad")
    parser.add_argument("--frame_gap", type=int, default=0)
    parser.add_argument("--use_wandb", type=str, default="True", help="True if you want to log to wandb")
    parser.add_argument("--log_freq", type=int, default=2, help="Log frequency")
    args = parser.parse_args()
    main(args)

