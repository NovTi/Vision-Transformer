from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image


class whale_dolphin(Dataset):
    def __init__(self, root, train=True):
        train_imgs = []
        val_imgs = []
        for path in os.listdir(root):
            label = int(path[:1])
            path_lst = path.split(".")
            if int(path_lst[0][2:]) < 1301:
                train_imgs.append((os.path.join(root, path), label))
            else:
                val_imgs.append((os.path.join(root, path), label))
        self.imgs = train_imgs if train else val_imgs
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        img_label = self.imgs[index][1]
        img_data = Image.open(img_path)
        img_data = self.transforms(img_data)
        return img_data, img_label


# This this the data loader for test set
class whale_dolphin_test(Dataset):
    def __init__(self, root):
        self.imgs = []
        for path in os.listdir(root):
            label = int(path[:1])
            path_lst = path.split(".")
            self.imgs.append((os.path.join(root, path), label))
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        img_label = self.imgs[index][1]
        img_data = Image.open(img_path)
        img_data = self.transforms(img_data)
        return img_data, img_label


if __name__ == '__main__':
    root = 'dataset/train'
    train_dataset = whale_dolphin(root, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for data, label in train_dataloader:
        print(data.shape)
        print(len(label))
        break
