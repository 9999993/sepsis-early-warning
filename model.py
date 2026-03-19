#!/usr/bin/env python3
"""
脓毒症早期预警模型 - LSTM深度学习架构。

本模块实现了基于长短期记忆网络(LSTM)的脓毒症预测模型，
旨在提供发病前12小时的早期预警。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SepsisDataset(Dataset):
    """
    PyTorch Dataset类用于脓毒症时间序列数据。

    将生成的数据转换为PyTorch可用的格式。

    属性:
        X: 特征数据数组
        y: 标签数组
        mask: 数据可用性掩码
    """

    def __init__(self, data_dict: Dict):
        """
        初始化数据集。

        参数:
            data_dict: 包含'X'、'y'、'mask'键的字典
        """
        self.X = torch.FloatTensor(data_dict["X"])
        self.y = torch.FloatTensor(data_dict["y"])
        self.mask = torch.FloatTensor(data_dict["mask"])

        logger.info(f"Initialized dataset: {len(self.X)} samples, shape {self.X.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本。

        参数:
            idx: 样本索引

        返回:
            元组(X, y, mask)
        """
        return self.X[idx], self.y[idx], self.mask[idx]


class AttentionMechanism(nn.Module):
    """
    自注意力机制用于增强LSTM的特征表达能力。

    通过计算不同时间步之间的相关性,模型可以关注
    对脓毒症预测最重要的生理参数变化。

    属性:
        hidden_dim: LSTM隐藏层维度
        attention_dim: 注意力层维度
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        """
        初始化注意力机制。

        参数:
            hidden_dim: LSTM隐藏状态的维度
            attention_dim: 注意力层的中间维度
        """
        super(AttentionMechanism, self).__init__()

        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim

        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算注意力权重。

        参数:
            lstm_output: LSTM输出,形状为(batch, seq_len, hidden_dim)

        返回:
            加权后的特征向量和注意力权重
        """
        # 计算注意力得分
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        attention_weights = self.attention(lstm_output)

        # 在序列维度上计算softmax
        # (batch, seq_len, 1) -> (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 应用注意力权重到LSTM输出
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> (batch, hidden_dim)
        context = torch.sum(lstm_output * attention_weights, dim=1)

        return context, attention_weights.squeeze(-1)


class SepsisNet(nn.Module):
    """
    脓毒症预警神经网络模型。

    架构包括:
    - 初始特征投影层
    - 双向LSTM层(多层)
    - 自注意力机制
    - 时间序列预测层

    设计用于学习时间序列中的长期依赖关系和临界模式。

    属性:
        input_dim: 输入特征维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        output_dim: 输出维度(时间步的分类)
        dropout: Dropout概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.3,
        use_attention: bool = True,
        bidirectional: bool = True,
    ):
        """
        初始化神经网络模型。

        参数:
            input_dim: 输入特征数量
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 每个时间步的输出维度
            dropout: Dropout比率
            use_attention: 是否使用注意力机制
            bidirectional: 是否使用双向LSTM
        """
        super(SepsisNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        # 输入投影层 - 将原始特征投影到高维空间
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # LSTM输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 注意力机制(可选)
        if use_attention:
            self.attention = AttentionMechanism(
                lstm_output_dim, attention_dim=hidden_dim // 2
            )

        # 时间步级别的分类层
        self.temporal_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 初始化权重
        self._init_weights()

        logger.info(
            f"Initialized SepsisNet: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"bidirectional={bidirectional}, attention={use_attention}"
        )

    def _init_weights(self):
        """使用Xavier初始化网络权重。"""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        参数:
            x: 输入数据,形状(batch, seq_len, input_dim)
            return_attention: 是否返回注意力权重

        返回:
            预测概率和(可选的)注意力权重
        """
        batch_size, seq_len, _ = x.shape

        # 输入投影
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        x = self.input_projection(x)

        # LSTM前向传播
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(x)

        # 为每个时间步生成预测
        # (batch, seq_len, lstm_dim) -> (batch, seq_len, 1)
        predictions = self.temporal_classifier(lstm_out)

        # 应用sigmoid得到概率
        probabilities = torch.sigmoid(predictions)

        if return_attention and self.use_attention:
            _, attention_weights = self.attention(lstm_out)
            return probabilities.squeeze(-1), attention_weights

        return probabilities.squeeze(-1), None

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        为输入数据预测概率。

        参数:
            x: 输入数据

        返回:
            预测概率数组
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)

            if x.dim() == 2:
                x = x.unsqueeze(0)

            probs, _ = self.forward(x)
            return probs.cpu().numpy()


class SepsisLoss(nn.Module):
    """
    脓毒症预测的自定义损失函数。

    结合了:
    - 加权二元交叉熵损失(处理类别不平衡)
    - 时间平滑惩罚(避免振荡预测)
    - 早期预警奖励(鼓励提前预测)
    """

    def __init__(
        self,
        pos_weight: float = 3.0,
        temporal_smooth_weight: float = 0.1,
        early_warning_weight: float = 0.2,
        early_warning_window: int = 12,
    ):
        """
        初始化损失函数。

        参数:
            pos_weight: 正样本(脓毒症)的权重
            temporal_smooth_weight: 时间平滑项的权重
            early_warning_weight: 早期预警奖励权重
            early_warning_window: 脓毒症前多少小时的预测奖励
        """
        super(SepsisLoss, self).__init__()

        self.pos_weight = pos_weight
        self.temporal_smooth_weight = temporal_smooth_weight
        self.early_warning_weight = early_warning_weight
        self.early_warning_window = early_warning_window

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权组合损失。

        参数:
            predictions: 模型预测概率, 形状 (batch, seq_len)
            targets: 真实标签, 形状 (batch, seq_len)
            mask: 数据有效性掩码, 形状 (batch, seq_len)

        返回:
            标量损失值
        """
        # 确保所有张量形状一致
        if mask.dim() == 3:
            mask = mask.squeeze(-1)

        valid_mask = mask

        # 1. 加权交叉熵损失
        # 计算正负样本权重
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        weights = pos_mask * self.pos_weight + neg_mask * 1.0

        # 二元交叉熵
        bce_loss = nn.functional.binary_cross_entropy(
            predictions, targets, reduction="none"
        )

        # 应用掩码并计算平均损失
        masked_bce = bce_loss * weights * valid_mask
        bce_loss = masked_bce.sum() / (valid_mask.sum() + 1e-8)

        # 2. 时间平滑损失 - 惩罚相邻时间步之间的剧烈变化
        if self.temporal_smooth_weight > 0:
            # 计算时间差分
            temporal_diff = predictions[:, 1:] - predictions[:, :-1]
            smooth_loss = torch.mean(temporal_diff**2)
        else:
            smooth_loss = torch.tensor(0.0, device=predictions.device)

        # 3. 早期预警奖励损失
        # 鼓励在脓毒症发病前的窗口期内给出高概率预测
        if self.early_warning_weight > 0:
            # 找到每个样本的脓毒症发病时间
            seizure_indices = torch.argmax(targets, dim=1)

            # 构建早期预警窗口掩码
            batch_size, seq_len = targets.shape
            early_warning_mask = torch.zeros_like(targets)

            for i in range(batch_size):
                seizure_time = seizure_indices[i]
                if targets[i, seizure_time] == 1:
                    # 窗口从12小时前到发病时
                    window_start = max(0, seizure_time - self.early_warning_window)
                    early_warning_mask[i, window_start : seizure_time + 1] = 1.0

            # 在窗口内预测较低的惩罚
            early_warning_loss = -torch.mean(predictions * early_warning_mask * targets)
        else:
            early_warning_loss = torch.tensor(0.0, device=predictions.device)

        # 总损失
        total_loss = (
            bce_loss
            + self.temporal_smooth_weight * smooth_loss
            + self.early_warning_weight * early_warning_loss
        )

        return total_loss


class FocalLoss(nn.Module):
    """
    焦点损失(Focal Loss) - 解决类别极度不平衡问题。

    通过降低易分类样本的权重,使模型专注于难分类样本。
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 2.0, pos_weight: float = 3.0):
        """
        初始化焦点损失。

        参数:
            alpha: 平衡因子
            gamma: 聚焦参数
            pos_weight: 正样本权重
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算焦点损失。

        参数:
            predictions: 预测概率
            targets: 真实标签
            mask: 数据掩码

        返回:
            损失值
        """
        # 计算BCE
        bce = nn.functional.binary_cross_entropy(predictions, targets, reduction="none")

        # 计算pt
        pt = torch.exp(-bce)

        # 计算权重
        pos_mask = (targets == 1).float()
        weights = pos_mask * self.pos_weight + (1 - pos_mask)

        # 焦点损失公式
        focal_loss = weights * self.alpha * (1 - pt) ** self.gamma * bce

        # 应用掩码并求平均
        valid_mask = mask if mask.dim() == predictions.dim() else mask.unsqueeze(-1)
        loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss


def load_model(model_path: str, device: str = "cpu") -> SepsisNet:
    """
    从文件加载训练好的模型。

    参数:
        model_path: 模型文件路径(.pth)
        device: 加载到的设备

    返回:
        加载的模型实例
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 读取模型配置
    config = checkpoint["model_config"]

    # 创建模型
    model = SepsisNet(**config)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model config: {config}")

    return model


def save_model(
    model: SepsisNet, model_path: str, additional_info: Optional[Dict] = None
):
    """
    保存模型到文件。

    参数:
        model: 要保存的模型
        model_path: 保存路径
        additional_info: 额外要保存的信息(如训练指标)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "output_dim": model.output_dim,
            "use_attention": model.use_attention,
            "bidirectional": model.bidirectional,
        },
        "save_timestamp": str(Path.cwd()),
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    # 简单的模型测试
    print("Testing SepsisNet model...")

    # 创建测试数据
    batch_size = 4
    seq_len = 72
    input_dim = 20

    model = SepsisNet(input_dim=input_dim, hidden_dim=128, num_layers=2)

    # 测试前向传播
    test_input = torch.randn(batch_size, seq_len, input_dim)
    predictions, attention = model(test_input, return_attention=True)

    print(f"Model output shape: {predictions.shape}")
    print(
        f"Attention weights shape: {attention.shape if attention is not None else None}"
    )

    # 测试预测
    probas = model.predict_proba(test_input[0])
    print(f"Single sample prediction shape: {probas.shape}")

    print("Model test completed successfully!")
