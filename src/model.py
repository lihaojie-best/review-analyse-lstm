import torch
from torch import nn
import config


class ReviewAnalyseModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 词表大小决定了 词向量的个数
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx  # 次表中<pad> 填充的索引，通过tokenizer的属性获取
        )
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.liner = nn.Linear(
            in_features=config.HIDDEN_SIZE,  # lstm的输出维度 也就是隐藏层lstm的输出维度
            out_features=1  # 二分类
        )

    def forward(self, x): # x shape [batch_size,seq_len]
        # embedding shape [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        #print(f"embedding(x) shape {embedded.shape}") # embedding shape [batch_size, seq_len, embedding_dim]
        output, (h_0,c_0), = self.lstm(embedded) # output shape [batch_size, seq_len, hidden_size]
        output = self.liner(output[:, -1, :]) # -1 表示取出最后一个时间步的输出   output shape [batch_size, 1]
        output = output.squeeze(dim=1)
        return  output # output shape [batch_size]
