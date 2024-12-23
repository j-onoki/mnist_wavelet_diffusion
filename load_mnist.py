from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

#データのロード
def load_MNIST():
    #訓練データ
    train_dataset = datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.Compose([transforms.ToTensor()]),
                                            download = True)
    #検証データ
    test_dataset = datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.Compose([transforms.ToTensor()]),
                                            download = True)
    return train_dataset, test_dataset

#ミニバッチの作成
def loader_MNIST(train_dataset, test_dataset):

    batch_size = 100

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    return train_loader, test_loader

#前処理用の関数
def conv(image, kernel):
    w, h = image.size()[0], image.size()[1]
    s = kernel.size()[0]
    output = torch.zeros((w//2, h//2))
    for i in range(w//2):
        for j in range(h//2):
            output[i, j] = torch.sum(image[i*2:i*2+2, j*2:j*2+2]*kernel)

    return output

def wt_ll(image):
    kll = torch.tensor([[0.25, 0.25],[0.25, 0.25]])
    output = conv(image[0], kll)
    return output.reshape(1, output.size()[0], output.size()[1])
