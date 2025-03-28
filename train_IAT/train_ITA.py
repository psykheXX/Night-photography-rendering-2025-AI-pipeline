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
from IAT import IAT
from IQA_pytorch import SSIM
from IAT import PSNR, validation, LossNetwork
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--val_data_path', type=str, default='D:/NTIRE_2025/validation1/pipeline_raw/')
parser.add_argument('--val_label_path', type=str, default="D:/NTIRE_2025/validation1/sony/")
parser.add_argument('--train_data_path', type=str, default='D:/NTIRE_2025/train/pipeline_raw/')
parser.add_argument('--train_label_path', type=str, default='D:/NTIRE_2025/train/sony/')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrain_dir', type=str, default='./workdirs3/best_Epoch16.pth')

parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="./workdirs4/")

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
    model = IAT().cuda()
    if config.pretrain_dir is not None:
        model.load_state_dict(torch.load(config.pretrain_dir))

    # Data Setting
    train_dataset = TrainDataset(patch_size=1024, stride=976, huawei_root=config.train_data_path,
                                 sony_root=config.train_label_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                               collwwqate_fn=collate_fn)
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
    L1_smooth_loss = F.smooth_l1_loss

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

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
    model.train()
    print('######## Start IAT Training #########')
    for epoch in range(config.num_epochs):
        # adjust_learning_rate(optimizer, epoch)
        print('the epoch is:', epoch)
        for iteration, (low_img, high_img) in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            mul, add, enhance_img = model(low_img)

            loss = L1_loss(enhance_img, high_img)
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