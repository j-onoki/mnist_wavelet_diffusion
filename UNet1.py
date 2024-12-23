import torch
import torch.nn as nn
import torch.nn.functional as f
import ddpm_function as df

#時間の位置符号化
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.dim = dim
    
    def forward(self, t):
        t = df.embeddings(t, self.dim)
        time_emb = self.time_mlp(t)
        return self.time_mlp(t)

#畳み込みブロック
class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding = 1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.time_mlp = TimeEmbedding(dim_out)
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding = 1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t):

        res = self.res_conv(x)
        x = self.block1(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(2)
        time_emb = time_emb.unsqueeze(3)
        x = x + time_emb
        x = self.block2(x)
        x = x + res

        return x

#ダウンサンプルブロック(1層分)
class DownSampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups):
        super().__init__()
        self.conv1 = ResNetBlock(dim_in, dim_out, groups)
        self.conv2 = ResNetBlock(dim_out, dim_out, groups)
        self.pool = nn.MaxPool2d(2, stride = 2)
        
    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        c = x
        x = self.pool(x)

        return x, c

#ダウンサンプリング
class DownSampling(nn.Module):
    def __init__(self, layers, dim_in0, dim_out0, groups):
        super().__init__()
        self.module = nn.ModuleList([])
        dim_in = dim_in0
        dim_out = dim_out0
        for i in range(layers):
            self.module.append(DownSampleBlock(dim_in, dim_out, groups))
            dim_in = dim_out
            dim_out *= 2

    def forward(self, x, t):
        cache = []
        for m in self.module:
            x, c = m(x, t)
            cache.append(c)

        return x, cache

#中間ブロック
class MidBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups):
        super().__init__()
        self.conv1 = ResNetBlock(dim_in, dim_out, groups)
        self.conv2 = ResNetBlock(dim_out, dim_out, groups)

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)

        return x

#アップサンプルブロック(1層分)
class UpSampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)
        self.conv1 = ResNetBlock(dim_in, dim_out, groups)
        self.conv2 = ResNetBlock(dim_out, dim_out, groups)

    def forward(self, x, t, c):
        x = self.upconv(x)
        x = torch.cat((x, c), dim = 1)
        x = self.conv1(x, t)
        x = self.conv2(x, t)
    
        return x

#アップサンプリング
class UpSampling(nn.Module):
    def __init__(self, layers, dim_in0, dim_out0, groups):
        super().__init__()
        self.module = nn.ModuleList([])
        dim_in = dim_in0
        dim_out = dim_out0
        for i in range(layers):
            self.module.append(UpSampleBlock(dim_in, dim_out, groups))
            dim_in = dim_out
            dim_out //= 2

    def forward(self, x, t, cache):
        for m in self.module:
            x = m(x, t, cache.pop())
        
        return x

#UNet
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downs = DownSampling(3, 1, 64, 8)
        self.mid = MidBlock(256, 512, 8)
        self.ups = UpSampling(3, 512, 256, 8)
        self.conv = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        x, cache = self.downs(x, t)
        x = self.mid(x, t)
        x = self.ups(x, t, cache)
        x = self.conv(x)
        
        return x