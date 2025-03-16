from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os
from torch.utils.data import DataLoader
class TrainDataset(Dataset):
    def __init__(self, patch_size=256, arg=False, stride=8, huawei_root="../data/train/raw/",
                 sony_root="../data/train/sony/"):
        self.patch_size = patch_size
        self.raws = []
        self.RGBs = []
        self.arg = arg
        h, w = 2000, 2000
        self.stride = stride

        self.patch_per_row = (w - patch_size) // stride + 1
        self.patch_per_col = (h - patch_size) // stride + 1
        self.patch_per_img = self.patch_per_row * self.patch_per_col

        self.huawei_root = huawei_root
        self.sony_root = sony_root

        huawei_names = os.listdir(self.huawei_root)

        for huawei_name in huawei_names:
            if huawei_name.lower().endswith('.jpg'):
                huawei_path = os.path.join(self.huawei_root, huawei_name)
                sony_name = huawei_name
                sony_path = os.path.join(self.sony_root, sony_name)

                huawei_img = cv2.imread(huawei_path, cv2.IMREAD_COLOR)
                huawei_img = cv2.cvtColor(huawei_img,cv2.COLOR_BGR2RGB)
                sony_img = cv2.imread(sony_path, cv2.IMREAD_UNCHANGED)
                sony_img = cv2.cvtColor(sony_img,cv2.COLOR_BGR2RGB)

                huawei_img = np.transpose(huawei_img, (2, 0, 1))  # Channel-first format
                sony_img = np.transpose(sony_img, (2, 0, 1))

                huawei_img = (huawei_img / 255).astype(np.float32)
                sony_img = (sony_img / 255).astype(np.float32)

                self.raws.append(huawei_img)
                self.RGBs.append(sony_img)
                print(f'Ntire2025 scene {huawei_name} is loaded.')

        self.img_num = len(self.raws)
        self.patch_total_num = self.patch_per_img * self.img_num

    def argument(self, img, rot_times, v_flip, h_flip):
        if rot_times > 0:
            img = np.rot90(img.copy(), rot_times, axes=(1, 2))
        if v_flip:
            img = img[:, :, ::-1].copy()
        if h_flip:
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        patch_size = self.patch_size

        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img
        h_idx = patch_idx // self.patch_per_row
        w_idx = patch_idx % self.patch_per_row

        h_start, h_end = h_idx * stride, h_idx * stride + patch_size
        w_start, w_end = w_idx * stride, w_idx * stride + patch_size
        huawei = self.raws[img_idx][:, h_start:h_end, w_start:w_end]
        sony = self.RGBs[img_idx][:, h_start:h_end, w_start:w_end]

        rot_times = random.randint(0, 3)
        v_flip = random.randint(0, 1)
        h_flip = random.randint(0, 1)
        if self.arg:
            huawei = self.argument(huawei, rot_times, v_flip, h_flip)
            sony = self.argument(sony, rot_times, v_flip, h_flip)

        return np.ascontiguousarray(huawei), np.ascontiguousarray(sony)

    def __len__(self):
        return self.patch_total_num

class Validation2Dataset(Dataset):
    def __init__(self, huawei_root="D:/NTIRE 2025/validation2/pipeline_raw/"):
        self.raws = []

        self.huawei_root = huawei_root

        huawei_names = os.listdir(self.huawei_root)

        for huawei_name in huawei_names:
            if huawei_name.lower().endswith('.png'):
                huawei_path = os.path.join(self.huawei_root, huawei_name)


                huawei_img = cv2.imread(huawei_path, cv2.IMREAD_COLOR)

                huawei_img = np.transpose(huawei_img, (2, 0, 1))  # Channel-first format

                huawei_img = (huawei_img / 255).astype(np.float32)

                self.raws.append(huawei_img)
                print(f'Ntire2025 scene {huawei_name} is loaded.')

        self.img_num = len(self.raws)

    def __getitem__(self, idx):
        huawei = self.raws[idx]
        return np.ascontiguousarray(huawei)

    def __len__(self):
        return self.img_num

class TestDataset(Dataset):
    def __init__(self, huawei_root="D:/NTIRE 2025/first_validation_data/pipeline_raw/",
                 sony_root="D:/NTIRE 2025/first_validation_data/sony/"):
        self.raws = []
        self.RGBs = []

        self.huawei_root = huawei_root
        self.sony_root = sony_root

        huawei_names = os.listdir(self.huawei_root)

        for huawei_name in huawei_names:
            if huawei_name.lower().endswith('.jpg'):
                huawei_path = os.path.join(self.huawei_root, huawei_name)
                sony_name = huawei_name
                sony_path = os.path.join(self.sony_root, sony_name)

                huawei_img = cv2.imread(huawei_path, cv2.IMREAD_COLOR)
                huawei_img = cv2.cvtColor(huawei_img, cv2.COLOR_BGR2RGB)
                sony_img = cv2.imread(sony_path, cv2.IMREAD_UNCHANGED)
                sony_img = cv2.cvtColor(sony_img, cv2.COLOR_BGR2RGB)

                huawei_img = np.transpose(huawei_img, (2, 0, 1))  # Channel-first format
                sony_img = np.transpose(sony_img, (2, 0, 1))

                huawei_img = (huawei_img / 255).astype(np.float32)
                sony_img = (sony_img / 255).astype(np.float32)

                self.raws.append(huawei_img)
                self.RGBs.append(sony_img)
                print(f'Ntire2025 scene {huawei_name} is loaded.')

        self.img_num = len(self.raws)

    def __getitem__(self, idx):
        huawei = self.raws[idx]
        sony = self.RGBs[idx]
        return np.ascontiguousarray(huawei), np.ascontiguousarray(sony)

    def __len__(self):
        return self.img_num

class LoadJustRawName(Dataset):
    def __init__(self, huawei_root="D:/NTIRE 2025/first_validation_data/pipeline_raw/"):
        self.raws = []
        self.names = []

        self.huawei_root = huawei_root

        huawei_names = os.listdir(self.huawei_root)

        for huawei_name in huawei_names:
            if huawei_name.lower().endswith('.png'):
                huawei_path = os.path.join(self.huawei_root, huawei_name)

                huawei_img = cv2.imread(huawei_path, cv2.IMREAD_COLOR)

                huawei_img = np.transpose(huawei_img, (2, 0, 1))  # Channel-first format

                huawei_img = (huawei_img / 255).astype(np.float32)

                self.raws.append(huawei_img)
                self.names.append(huawei_name)
                print(f'Ntire2025 scene {huawei_name} is loaded.')

        self.img_num = len(self.raws)

    def __getitem__(self, idx):
        huawei = self.raws[idx]
        name = self.names[idx]
        return np.ascontiguousarray(huawei), name

    def __len__(self):
        return self.img_num

if __name__ == '__main__':
    # 创建数据集实例并展示数据
    dataset = TestDataset()

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (huawei_patch, sony_patch) in enumerate(dataloader):
        print(huawei_patch.shape, sony_patch.shape)

