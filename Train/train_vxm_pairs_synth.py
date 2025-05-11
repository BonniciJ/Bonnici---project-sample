from FYP_package.voxelmorph.voxelmorph_standard.models.networks import VxmDense
from FYP_package.voxelmorph.voxelmorph_standard.models.losses import NCC, MSE, Grad
from FYP_package.voxelmorph.voxelmorph_timeseries.models.losses import BendingEnergy

from FYP_package.voxelmorph.voxelmorph_timeseries.dataloaders.npy_time_series_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

import time
import argparse
import json

import wandb

# some functions for saving the model 
def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def save_and_log_images(fixed, moving, flo, warped, epoch, batch_size, save_name="train_figure"):
    # # Create the data folder if it doesn't exist
    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)

    num_imgs = min(batch_size, 16)
 
    fig, axes = plt.subplots(num_imgs, 5, figsize=(10, 3*num_imgs))
 
    for b in range(num_imgs):
        i = b
        j = 0
        img = fixed[i, 0].cpu().detach().numpy()      
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')     
        axes[i, j].set_title("Fixed")   
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = b
        j = 1
        img = moving[i, 0].cpu().detach().numpy()      
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')     
        axes[i, j].set_title("Moving")   
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 2
        img = flo[i, 0].cpu().detach().numpy()      
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("Displacement X")    
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 3
        img = flo[i, 1].cpu().detach().numpy()     
        im = axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')    
        axes[i, j].set_title("Displacement Z")
        fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)# Add colorbar

        i = i
        j = 4
        img = warped[i, 0].cpu().detach().numpy()      
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

    torch.manual_seed(0)
    np.random.seed(0)

    if args.use_wandb  == "True":
        # Initialize wandb 
        wandb.init(project="voxelmorph_run_timeseries")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    print()
    print(args)    

    train_dataset = TimeSeriesDataset(args.data_path, n_frames=2, skip_frames=[0]) #this gives pairs

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
 
    print(f"Train size: {len(train_dataset)}")



    if args.use_Bending_loss == "True":
        smoothness_loss = BendingEnergy()
        print("Using bending energy regularisation")
    else:
        smoothness_loss = Grad()
        print("Using grad regularisation")
        
    if args.use_NCC_loss == "True":
        similarity_loss = NCC()
        print("Using NCC sim Loss")
    else:
        similarity_loss = MSE()
        print("Using MSE sim Loss")
   
    model = VxmDense(
        inshape=(args.image_height, args.image_width), 
        int_downsize=1, 
        nb_unet_features=args.nb_features, 
    )

    # Support for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 

    # Load from old 
    # checkpoint = torch.load("savedModels/model_best.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1
    
    

    #check save folder exists
    #save_checkpoint(model, optimizer, 0, filepath=f"savedModels\model_best.pth")

    #best_val_loss = np.inf
    start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs):

        # ------------------- Training loop -------------------+
        print()
        print(f"Epoch {epoch}")
        print("------------")
        
        model.train()
        epoch_loss = 0
        epoch_smooth_loss = 0
        epoch_sim_loss = 0

        t0, t1, t2 = 0, 0, 0

        t04 = time.perf_counter()      
        for batch_idx, frames in enumerate(train_loader):   
            t00 = time.perf_counter()      
            fixed = frames[:, 0, :, :].unsqueeze(1).float().to(device, non_blocking=True) # first frame
            moving = frames[:, 1, :, :].unsqueeze(1).float().to(device, non_blocking=True) # second frame
            t01 = time.perf_counter()

            # Forward pass
            warped, displacement_field = model(moving, fixed) 
            t02 = time.perf_counter()

            # Compute loss (NCC + smoothness)
            smooth_loss = smoothness_loss.loss(fixed, displacement_field)
            sim_loss = similarity_loss.loss(fixed, warped)
            loss = (1 - args.gamma) * sim_loss + args.gamma * smooth_loss
            t03 = time.perf_counter()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t0 += t00 - t04
            t1 += t02 - t01
            t2 += t03 - t02

            epoch_loss += loss.item()
            epoch_smooth_loss += smooth_loss.item()
            epoch_sim_loss += sim_loss.item()

            # Log each 50 batches
            if batch_idx % 50 == 0 and batch_idx > 1:  
                print(f"Epoch {epoch}:  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {(epoch_loss / (batch_idx+1)):.2f}, smoothL: {(epoch_smooth_loss / (batch_idx+1)):.2f}, simL: {(epoch_sim_loss / (batch_idx+1)):.2f}")   
                print(f"Time to load data: {t0:.5f}, Time to forward pass: {t1:.5f}, Time to compute loss: {t2:.5f}")
                print()

                if args.use_wandb == "True":
                    # log metrics
                    wandb.log({
                        "train_loss": epoch_loss / (batch_idx+1),
                        "smooth_loss": epoch_smooth_loss / (batch_idx+1),
                        "sim_loss": epoch_sim_loss / (batch_idx+1),
                        "epoch": epoch,
                    })
                t0, t1, t2 = 0, 0, 0 

            if batch_idx % 500 == 0:  
                if args.use_wandb == "True":
                    save_and_log_images(fixed, moving, displacement_field, warped, epoch, args.batch_size, save_name="mid_epoch_fig")

            t04 = time.perf_counter()      

        # Print training loss
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train loss: {epoch_loss}")
        if args.use_wandb == "True":
            wandb.log({
                        "Epoch_loss": epoch_loss,
                    })
            
        
        save_checkpoint(model, optimizer, epoch, filepath=f'{args.checkpoint_path}/chpt_epoch_{epoch}.pth')
        print(f'{args.checkpoint_path}/chpt_epoch_{epoch}.pth')
            

    if args.use_wandb == "True":
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="D:/movies_npy/train", help="Path to the dataset")
    parser.add_argument("--n_frames", type=int, default=3, help="Number of frames in timeseries")
    parser.add_argument("--skip_frames", nargs="+", type=int, default=[0, 1, 2, 3], help="Skips in frames between each frame in timeseries (changes frame rate)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weighting of smoothness loss (0=no smoothing)")
    #parser.add_argument("--nb_features", nargs="+", type=int, default=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]], help="Number of features in U-Net encoder and decoder")
    parser.add_argument(
    "--nb_features",
    type=str,
    default='[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]',
    help="JSON string for number of features in U-Net encoder and decoder"
)
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--image_height", type=int, default=320, help="Image height")
    parser.add_argument("--image_width", type=int, default=320, help="Image width")
    parser.add_argument("--save_model", action="store_true", help="Flag to save the model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="hearts_pairs_model", help="Path to save the model checkpoint")
    parser.add_argument("--use_NCC_loss", type=str, default="True", help="Use NCC loss if true, otherwise use MSE")
    parser.add_argument("--use_Bending_loss", type=str, default="True", help="Use Bending energy if true, otherwise use Grad")
    parser.add_argument("--use_wandb", type=str, default="True", help="True if you want to log to wandb")
    args = parser.parse_args()
    args.nb_features = json.loads(args.nb_features)
    main(args)


