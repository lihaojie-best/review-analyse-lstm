import pandas as pd
from numpy.core import records

import config
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer

def process():
    print("数据处理开始")
    # 读取文件
    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv", encoding="utf-8",usecols=["review","label"])
    # 过滤数据
    df = df.dropna()
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    # 创建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(),config.PROCESSED_DATA_DIR / 'vocab.txt')
    # tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(text=x,seq_len=config.SEQ_LEN))
    # 保存训练集
    train_df.to_json(config.PROCESSED_DATA_DIR/'indexed_train.jsonl',orient="records",lines=True)
    # 构建测试集
    test_df['review'] = test_df['review'].apply(lambda x:tokenizer.encode(text=x,seq_len=config.SEQ_LEN))
    # 保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR/'indexed_test.jsonl',orient="records",lines=True)
    print("数据处理结束")





if __name__ == '__main__':
    process()

