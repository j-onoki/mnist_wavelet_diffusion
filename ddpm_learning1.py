import torch
import ddpm_function as df
from tqdm import tqdm

#1epoch分の訓練を行う関数
def train_model(model, train_loader, optimizer, device):

    train_loss = 0.0

    # 学習モデルに変換
    model.train()


    for i, (images, labels) in enumerate(tqdm(train_loader)):

        images, labels = images.to(device), labels.to(device)
        images = images.unsqueeze(1)

        images, t, epsilon = df.addNoise(images, device)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        outputs = model(images, t)

        # 損失の算出
        loss = df.criterion(outputs.reshape(100,8,8), epsilon.to(device))
        
        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    
    # lossの平均値を取る
    train_loss = train_loss / len(train_loader)
    
    return train_loss

#モデル評価を行う関数
def test_model(model, test_loader, optimizer, device):

    test_loss = 0.0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化

        for i, (images, labels) in enumerate(tqdm(test_loader)):

            images, labels = images.to(device), labels.to(device)
            images = images.unsqueeze(1)#reshape(100, 1, 8, 8)
            images, t, epsilon = df.addNoise(images, device)

            outputs = model(images, t)
            loss = df.criterion(outputs.reshape(100,8,8), epsilon.to(device))
            test_loss += loss.item()


    # lossの平均値を取る
    test_loss = test_loss / len(test_loader)

    return test_loss

def lerning(model, train_loader, test_loader, optimizer, num_epochs, device):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss = train_model(model, train_loader, optimizer, device)
        test_loss = test_model(model, test_loader, optimizer, device)
        
        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    
    return train_loss_list, test_loss_list
