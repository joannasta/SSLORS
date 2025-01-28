import pytorch_lightning as pl
import numpy as np
import scipy
from torch.utils.data import DataLoader
from .mbn_dataset import MagicBathyNetDataset
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

class MagicBathyNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, modality, batch_size=32, transform=None, cache=False):
        super().__init__()
        self.root_dir = root_dir
        self.modality = modality
        self.batch_size = batch_size
        self.transform = transform
        self.cache = cache
        self.norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
        self.norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])

        # Load common model parameters
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]

    def setup(self, stage=None):
        # Initialize train and test datasets
        self.train_dataset = MagicBathyNetDataset(
            root_dir=self.root_dir,
            modality=self.modality,
            transform=self.transform,
            split_type='train',
            cache=self.cache
        )
        
        self.test_dataset = MagicBathyNetDataset(
            root_dir=self.root_dir,
            modality=self.modality,
            transform=self.transform,
            split_type='test',
            cache=self.cache
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

'''
import torch
import os
import matplotlib.pyplot as plt
def test_dataset():
    root_dir = "/faststorage/joanna/magicbathynet/MagicBathyNet"  # Update with correct path

    # Define train and test image lists
    train_images = ['409', '418', '350', '399', '361', '430', '380', '359', 
                    '371', '377', '379', '360', '368', '419', '389', '420', 
                    '401', '408', '352', '388', '362', '421', '412', '351', 
                    '349', '390', '400', '378']
    
    test_images = ['411', '387', '410', '398', '370', '369', '397']

    # Create data module instance
    data_module = MagicBathyNetDataModule(root_dir=root_dir, modality='s2', batch_size=4)  

    # Initialize data module

    data_module.setup()  # Set up data
    crop_size = 256
    WINDOW_SIZE = (256, 256)
    norm_param_depth = -30.443
    ratio = crop_size / WINDOW_SIZE[0]

    # Load train dataloader
    train_loader = data_module.train_dataloader()
    norm_param = np.load('/faststorage/joanna/magicbathynet/MagicBathyNet/agia_napa/norm_param_s2_an.npy')

    # Test data loading and plotting
    for batch_idx, (images, depths) in enumerate(train_loader):
        
        for (img, depth) in zip(images, depths):
            # Plot the pixel distribution (histogram) for normalized values
            plt.figure(figsize=(10, 6))
            plt.hist(img.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title("Pixel Normalized Value Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.savefig("pixel_normalized_distribution.png")  # Save the histogram plot
            plt.show()
            
            plt.figure(figsize=(10, 6))
            plt.hist(depth.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title(" Depth Pixel Normalized Value Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.savefig("depth_pixel_normalized_distribution.png")  # Save the histogram plot
            plt.show()
            

            # Denormalize the data (using updated formula)
            #img = img * (norm_param[1][:, np.newaxis, np.newaxis]) + norm_param[0][:, np.newaxis, np.newaxis]

            #Normalize the image to the range [0, 1] for displaying
            #denormalized_img = np.clip(img, 0, 255)  # Clip values to [0, 255] range
            #denormalized_img = denormalized_img / 255.0  # Normalize to [0, 1] if needed
            #depth = depth * norm_param_depth
            #depth = scipy.ndimage.zoom(depth, (1, 1 / ratio, 1 / ratio), order=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(depth.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title(" Depth Pixel Denormalized Value Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.savefig("depth_pixel_denormalized_distribution.png")  # Save the histogram plot
            plt.show()
            

            # Plot the images and depth maps
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            processed_img = np.transpose(img, (1, 2, 0))  # Convert from (3, 256, 256) to (256, 256, 3)

            
            axes[0].imshow(processed_img[:,:,[2,1,0]]) 
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(depth)  # Depth map visualization
            axes[1].set_title('Depth Map')
            axes[1].axis('off')

            # Save the figure as an image file
            save_path = f"batch_{batch_idx}_images.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
            plt.close(fig)

            if batch_idx > 2:  # Run for 3 batches for a quick test
                break

if __name__ == '__main__':
    test_dataset() '''
