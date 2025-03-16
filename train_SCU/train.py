import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import os
import sys
sys.path.append('..')
import argparse
from torchvision.models import vgg16
from IAT import TrainDataset, TestDataset
from IQA_pytorch import SSIM
from IAT import PSNR, validation, LossNetwork
import wandb
from SCUNet import SCUNet as net

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--val_data_path', type=str, default='D:/NTIRE_2025/validation1/enhanced2_img/')
parser.add_argument('--val_label_path', type=str, default="D:/NTIRE_2025/validation1/sony/")
parser.add_argument('--train_data_path', type=str, default='D:/NTIRE_2025/train/enhanced2_img/')
parser.add_argument('--train_label_path', type=str, default='D:/NTIRE_2025/train/sony/')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrain_dir', type=str, default='./workdirs3/best_Epoch4.pth')

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="./workdirs4/")
parser.add_argument('--model_name', type=str, default='scunet_color_real_psnr',
                    help='scunet_color_real_psnr, scunet_color_real_gan')
def gaussian_kernel(size=5, sigma=1.0, channels=3):
    """ 生成高斯核 """
    x_coord = torch.arange(size) - size // 2
    y_coord = torch.arange(size) - size // 2
    x_grid, y_grid = torch.meshgrid(x_coord, y_coord, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def gaussian_blur(image, kernel):
    """ 对图像应用高斯模糊 """
    padding = kernel.shape[-1] // 2
    return F.conv2d(image, kernel.to(image.device), padding=padding, groups=image.shape[1])

def collate_fn(batch):
    low_img, high_img = zip(*batch)
    # 确保数据是 PyTorch Tensor 类型
    low_img = torch.stack([torch.tensor(img) for img in low_img]).cuda()
    high_img = torch.stack([torch.tensor(img) for img in high_img]).cuda()
    return low_img, high_img

if __name__ == '__main__':
    config = parser.parse_args()

    print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    # Model Setting
    n_channels = 3

    model_path = config.pretrain_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.to(device)

    model.load_state_dict(torch.load(model_path), strict=True)

    # Data Setting
    train_dataset = TrainDataset(patch_size=512, stride=496, huawei_root=config.train_data_path,
                                 sony_root=config.train_label_path, arg=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                               collate_fn=collate_fn)
    val_dataset = TestDataset(huawei_root=config.val_data_path, sony_root=config.val_label_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                             collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-4,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    device = next(model.parameters()).device
    print('the device is:', device)

    # Loss & Optimizer Setting & Metric
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()

    for param in vgg_model.parameters():
        param.requires_grad = False

    # L1_loss = CharbonnierLoss()
    L1_loss = nn.L1Loss()
    L1_loss.to(device)
    L1_smooth_loss = F.smooth_l1_loss
    loss_color = F.mse_loss

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    loss_network.to(device)

    ssim = SSIM()
    psnr = PSNR()
    ssim_high = 0
    psnr_high = 0

    api_key = "3b906bb46b6ec6d959e620ee8f495f9fe62d5ebb"
    wandb.login(key=api_key)
    wandb.init(
        project="AWB",
        config={
            "model": "IAT",
            "batch_size": 4,
            "optimizer": "AdamW",
        }
    )
    print('######## Start IAT Training #########')
    for epoch in range(config.num_epochs):
        # adjust_learning_rate(optimizer, epoch)
        print('the epoch is:', epoch)
        model.train()
        for k, v in model.named_parameters():
            v.requires_grad = True
        for iteration, (low_img, high_img) in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            enhance_img = model(low_img)

            kernel = gaussian_kernel(size=5, sigma=1.0)
            enhance_img_g = gaussian_blur(enhance_img, kernel)
            high_img_g = gaussian_blur(high_img, kernel)
            loss = L1_loss(enhance_img, high_img) + 0.5 * loss_color(enhance_img_g, high_img_g)
            # loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
            loss.backward()

            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
                wandb.log({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
        scheduler.step()
        # Evaluation Model
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        PSNR_mean, SSIM_mean = validation(model, val_loader)

        with open(config.snapshots_folder + '/log.txt', 'a+') as f:
            f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')
            print(epoch, PSNR_mean, SSIM_mean)
            wandb.log({
                'SSIM': SSIM_mean,
                'PSNR': PSNR_mean
            })
        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            print('the highest SSIM value is:', str(ssim_high))
            torch.save(model.state_dict(), os.path.join(config.snapshots_folder, f"best_Epoch{epoch}" + '.pth'))
            print('the highest SSIM value is:', str(ssim_high))

    f.close()
    wandb.finish()