import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from dataloader import whale_dolphin
from PatchEncoding import PatchEncoding
from TransformerEncoder import TransformerEncoder
from TransformerEncoder import SelfAttention
from ClassificationHead import ClassificationHead
from ViT import ViT


def main():
    # 1. load dataset
    root = 'dataset/train'
    batch_size = 64
    train_data = whale_dolphin(root, train=True)
    val_data = whale_dolphin(root, train=False)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # 2. Prepare the sub-models
    Parts = [PatchEncoding, SelfAttention, TransformerEncoder, ClassificationHead]

    # 3.load model
    num_classes = 2
    img_channels = 3
    img_size = 128
    heads_num = 8
    patch_size = 32
    model = ViT(Parts, img_channels, patch_size, img_size, num_classes, heads_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
    model = model.to(device)

    # 4.prepare hyperparameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 20

    # 5.train
    val_acc_list = []
    out_dir = "results/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))

        print("Waiting Val...")
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Val\'s ac is: %.3f%%' % (100 * correct / total))

            acc_val = 100. * correct / total
            val_acc_list.append(acc_val)

        torch.save(model.state_dict(), out_dir + "last.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print(f"save epoch {epoch} model")


if __name__ == '__main__':
    main()
