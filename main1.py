import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import UNet1
import load_mnist as l
import ddpm_learning1
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #GPUが使えるか確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    #モデルのインスタンス化
    model = UNet1.UNet().to(device)
    print(model)

    #MNISTデータのダウンロード
    train_images = torch.load("./data_wt/train_ll2.pt")
    test_images = torch.load("./data_wt/test_ll2.pt")
    train_labels = torch.load("./data_wt/train_label.pt")
    test_labels = torch.load("./data_wt/test_label.pt")
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    #ミニバッチの作成
    train_loader, test_loader = l.loader_MNIST(train_dataset, test_dataset)

    #最適化法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3000
    train_loss_list, test_loss_list = ddpm_learning1.lerning(model, train_loader, test_loader, optimizer, num_epochs, device)

    plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('./image/loss1.png')

    #モデルを保存する。
    torch.save(model.state_dict(), "model1.pth")