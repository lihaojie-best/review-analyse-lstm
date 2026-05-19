import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from src.model import ReviewAnalyseModel
from src.tokenizer import JiebaTokenizer
from dataset import get_dataloader


def train_one_epoch(model, dataloader, loss_function, optimize, device):
    # 将模型设置为训练模式
    model.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(dataloader,desc='训练'):
        # inputs shape [batch_size,seq_len]
        # targets shape [batch_size]
        # 数据移动到设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 梯度清零
        optimize.zero_grad()
        # 前向传播
        outputs = model(inputs)  # outputs shape [batch_size,1]
        # 损失函数
        loss = loss_function(outputs, targets)
        # 反向传播
        loss.backward()
        # 更新参数
        optimize.step()
        # 统计损失
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 导入词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "vocab.txt")
    # 创建模型
    model = ReviewAnalyseModel(tokenizer.vocab_size, tokenizer.pad_token_id)
    # 模型移动到设备
    model.to(device)
    # 数据
    dataloader = get_dataloader(train=True)
    # 训练
    loss_function = nn.BCEWithLogitsLoss()
    # 优化器
    optimize = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 使用tensorBoard 记录损失函数
    writer = SummaryWriter(
        config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S"))  # time.strftime("%Y-%m-%d-%H-%M-%S") 引入时间子目录
    # 训练模型
    best_loss=float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"第{epoch}轮训练开始")
        mean_loss = train_one_epoch(model, dataloader, loss_function, optimize, device)
        print(f"平均损失{mean_loss:4f}")
        writer.add_scalar('Loss', mean_loss, epoch)
        if mean_loss<best_loss:
            best_loss=mean_loss
            torch.save(model.state_dict(),config.MODELS_DIR / 'models.pt')
            print("保存模型")
        else:
            print("模型未更新")


    writer.close()


if __name__ == '__main__':
    train()
