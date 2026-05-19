import torch

from src import config
from src.model import ReviewAnalyseModel
from src.tokenizer import JiebaTokenizer

def predict_batch(input_tensor,model):
    """
     批量预测
    :param input_tensor: 输入张量
    :param model: 模型
    :return: 一批的预测结果 [0.2,0.4,0.5...]
    """
    model.eval() # 模型设置为评估模式
    with torch.no_grad():
        output = model(input_tensor) # output shape [batch_size]
        # print(f"output{output}")
        # print(torch.sigmoid(output))
        # 调用激活函数 并转为列表
        return torch.sigmoid(output).tolist()


def predict(user_input, model, tokenizer, device):
    # 编码并填充输入文本
    index_list = tokenizer.encode(text=user_input, seq_len=config.SEQ_LEN)
    # 转换为张量并移动到设备
    input_tensor = torch.tensor([index_list]).to(device) # input tensor shape [1,seq_len]
    # 获取预测概率
    batch_result = predict_batch(input_tensor, model)
    return batch_result[0]


def run_predict():

    # 准备资源
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # Tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "vocab.txt")
    # models
    model = ReviewAnalyseModel(tokenizer.vocab_size,tokenizer.pad_token_id)
    # 模型加载参数
    model.load_state_dict(torch.load(config.MODELS_DIR / "models.pt"))
    model.to(device)
    print("请输入评论：（q or quit 退出）")
    while True:
        user_input = input("> ")
        if user_input in   ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue
        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f"positive（置信度：{result:.4f}）")
        else:
            print(f"negative（置信度：{result:.4f}）")


if __name__ == '__main__':
    run_predict()
