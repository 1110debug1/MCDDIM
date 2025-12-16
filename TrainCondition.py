import os
import random
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer, DDIMSampler, GaussianDiffusionSampler
from DiffusionFreeGuidence.ModelCondition1 import UNet
from Scheduler import GradualWarmupScheduler



class MyDataset3d(Dataset):
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.path_dir, image_index)
        img = np.load(img_path)
        img = np.array(img, dtype=np.float32).reshape(64, 64, 64)
        label = img_path.split('\\')[-1]
        label = label.strip('npy')[:-1]
        labela = np.float32(label.split('_')[0])
        labelb = np.float32(label.split('_')[1])
        label_pore = np.float32([labela, labelb])
        labelc = np.float32(label.split('_')[2])
        labeld = np.float32(label.split('_')[3])
        label_throat = np.float32([labelc, labeld])
        labele = np.float32(label.split('_')[4])
        labelf = np.float32(label.split('_')[5])
        label_coord = np.float32([labele, labelf])
        # label = np.float32(label)
        return img, label_pore, label_throat, label_coord

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Dataset setup
    dataset = MyDataset3d(modelConfig["path"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # Model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight loaded.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], betas=(0.9, 0.999), weight_decay=1e-4)

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    scaler = GradScaler()
    epoch_losses = []

    for e in range(modelConfig["epoch"]):
        epoch_loss = 0
        num_batches = 0

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i, (images, label2, label3, label4) in enumerate(dataloader):
                # Training process
                x_0 = images.to(device).type(torch.float32)
                labels = torch.sum(x_0 == 0, dim=(-1, -2, -3)) / (x_0.shape[-1] ** 3)
                x_0 = torch.unsqueeze(x_0, dim=1)  # Adding a channel dimension for 3D data
                label2 = label2.to(device)
                label3 = label3.to(device)
                label4 = label4.to(device)

                optimizer.zero_grad()
                with autocast():
                    # Forward pass through trainer, loss computation
                    loss = trainer(x_0, labels, label2, label3, label4).sum() / 1000.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Track loss
                epoch_loss += loss.item()
                num_batches += 1

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

                # # Periodically save noisy images (e.g., every 100 batches)
                # if i % 100 == 0:
                #     t = torch.randint(0, modelConfig["T"], (1,)).item()  # Pick a random timestep
                #     noisy_image = trainer.q_sample(x_0, t)  # Add noise at timestep t
                #
                #     # Convert to NumPy and save as .npy
                #     noisy_image_np = noisy_image.squeeze().cpu().numpy()  # Remove batch and channel dimensions
                #     np.save(f'{modelConfig["save_dir"]}/noisy_img_epoch{e}_batch{i}_t{t}.npy', noisy_image_np)

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        # Step the scheduler
        warmUpScheduler.step()

        # Save model if average loss is less than 100 or every 50 epochs
        if avg_epoch_loss < 20 or e % 50 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_dir"], f'berea_epoch_{e}_loss_{avg_epoch_loss:.2f}.pt'))
            print(f"Model saved at epoch {e} with average loss {avg_epoch_loss:.2f}")

        print(f"Epoch {e} completed with average loss: {avg_epoch_loss:.2f}")

    # Save final model
    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_dir"], f'final_model_after_{modelConfig["epoch"]}_epochs.pt'))

    # Save epoch and loss values to a CSV file
    df = pd.DataFrame({
        'Epoch': range(len(epoch_losses)),
        'Average Loss': epoch_losses
    })
    csv_path = os.path.join(modelConfig["save_dir"], 'epoch_loss.csv')
    df.to_csv(csv_path, index=False)
    print(f"Epoch and loss data saved to {csv_path}")

    # Plot the final loss curve with detailed x and y axis range
    plt.figure()
    plt.plot(range(len(epoch_losses)), epoch_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Total')

    # Set more detailed axis limits for better visualization
    plt.xlim(0, len(epoch_losses))  # X-axis from 0 to total epochs
    plt.ylim(min(epoch_losses) * 0.9, max(epoch_losses) * 1.1)  # Y-axis from a bit below min to a bit above max loss

    plt.grid(True)

    # Save the plot locally as 'loss_curve.png'
    loss_curve_path = os.path.join(modelConfig["save_dir"], 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.show()
    print(f"Loss curve saved as {loss_curve_path}")


def eval(modelConfig: Dict):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(90)
    # setup_seed(int(time.time()))  # 动态设置随机种子
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        labels = torch.empty(modelConfig["batch_size"], dtype=torch.float32).uniform_(modelConfig["porosity"] - 0.01,
                                                                                      modelConfig[
                                                                                          "porosity"] + 0.01).to(device)
        labela = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["pore_mean"] - 0.01,
            modelConfig["pore_mean"] + 0.01).to(device)
        labelb = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["por_std"] - 0.01,
            modelConfig["por_std"] + 0.01).to(device)
        label2 = torch.cat((labela, labelb), 1)

        labelc = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["throat_mean"] - 0.01,
            modelConfig["throat_mean"] + 0.01).to(device)
        labeld = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["throat_std"] - 0.01,
            modelConfig["throat_std"] + 0.01).to(device)
        label3 = torch.cat((labelc, labeld), 1)

        labele = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["coord_mean"] - 0.01,
            modelConfig["coord_mean"] + 0.01).to(device)
        labelf = torch.empty((modelConfig["batch_size"], 1), dtype=torch.float32).uniform_(
            modelConfig["coord_std"] - 0.01,
            modelConfig["coord_std"] + 0.01).to(device)
        label4 = torch.cat((labele, labelf), 1) #消融配位
        print("labels: ", labels, label2, label3, label4)


        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt, strict=False)
        # print("model load weight done.")
        model.eval()
        dir = 'D:/Pycharm_Professional/MCDDPM-main/npydata/condition/'
        # sampler = GaussianDiffusionSampler(
        #     model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        sampler = DDIMSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], T=350).to(device)

        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 1, 64, 64, 64], device=device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)

        sampledImgs = sampler(noisyImage, labels, label2, label3, label4)


        # 统计最大值、最小值
        max_value = torch.max(sampledImgs).item()
        min_value = torch.min(sampledImgs).item()
        print(f"Max value: {max_value}, Min value: {min_value}")

        # sampledImgs1 = (sampledImgs+1) / 2  # [0 ~ 1] 10.27修改
        # sampledImgs1 = sampledImgs * 0.5 + 0.5
        sampledImgs1 = torch.clamp(sampledImgs, 0, 1)

        # 统计最大值、最小值
        max_value = torch.max(sampledImgs1).item()
        min_value = torch.min(sampledImgs1).item()
        print(f"Max value: {max_value}, Min value: {min_value}")

        for i, j in enumerate(sampledImgs1):
            j = torch.squeeze(j, dim=1).cpu().numpy()
            np.save(dir + str(i) + str('.npy'), j)

        # save_3d(1 - sampledImgs1, modelConfig["nrow"], dir + 'train2_' + str(modelConfig["label"]) + ".tif")
