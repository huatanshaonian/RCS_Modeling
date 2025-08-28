"""
Autoencoder模型架构定义模块
包含标准自编码器、变分自编码器和增强自编码器的网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAutoencoder(nn.Module):
    """
    标准自编码器网络架构
    用于RCS数据的非线性降维
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None):
        """
        初始化标准自编码器

        参数:
        input_dim: 输入维度（RCS数据点数量）
        latent_dim: 潜在空间维度（类似POD中的主成分数量）
        hidden_dims: 隐藏层维度列表，默认为渐进式降维
        """
        super(StandardAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 如果未指定隐藏层维度，使用默认的渐进式结构
        if hidden_dims is None:
            hidden_dims = [4096, 2048, 1024, 512, 256]

        # 编码器结构（从输入维度逐步压缩到潜在维度）
        encoder_layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1 if i < len(hidden_dims)-1 else 0.05)  # 减少最后层的dropout
            ])
            prev_dim = hidden_dim

        # 最终编码层
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器结构（从潜在维度逐步重构到输入维度）
        decoder_layers = []
        prev_dim = latent_dim

        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05 if i < len(hidden_dims)-1 else 0.1)
            ])
            prev_dim = hidden_dim

        # 最终重构层
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """编码：输入->潜在表示"""
        return self.encoder(x)

    def decode(self, z):
        """解码：潜在表示->重构输出"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播：完整的编码-解码过程"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalAutoencoder(nn.Module):
    """
    变分自编码器(VAE)网络架构
    在标准自编码器基础上加入概率建模，用于更好的数据生成和插值
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None):
        """
        初始化VAE

        参数:
        input_dim: 输入维度
        latent_dim: 潜在空间维度
        hidden_dims: 编码器/解码器隐藏层维度
        """
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256, 128]

        # 编码器（输出均值和方差）
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # 解码器
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """编码：返回潜在变量的均值和对数方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从高斯分布采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码：潜在变量->重构输出"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


class ResidualBlock(nn.Module):
    """残差块模块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class EnhancedAutoencoder(nn.Module):
    """增强的自编码器，使用残差块"""
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()

        # 更深的编码器
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )

        # 多个残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(4096),
            ResidualBlock(4096),
        ])

        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True)
        )

        # 残差块（解码器）
        self.decode_res_blocks = nn.ModuleList([
            ResidualBlock(4096),
            ResidualBlock(4096),
        ])

        self.output_proj = nn.Linear(4096, input_dim)

    def encode(self, x):
        """编码过程"""
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.encoder(x)

    def decode(self, z):
        """解码过程"""
        x = self.decoder(z)
        for block in self.decode_res_blocks:
            x = block(x)
        return self.output_proj(x)

    def forward(self, x):
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数：重构损失 + KL散度损失

    参数:
    recon_x: 重构输出
    x: 原始输入
    mu: 潜在变量均值
    logvar: 潜在变量对数方差
    beta: KL损失权重（β-VAE）
    """
    # 重构损失（MSE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss