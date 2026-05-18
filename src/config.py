from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'


# 超参数
SEQ_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
