# frill从音频提取特征矩阵(time,2048)
import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_hub as hub


# 数据集路径
SAVE_DIR = "/mnt/p-wangzixv/data_wav/emo_data/"
TRAIN_DATASET_PATH = os.path.join(SAVE_DIR, "train_dataset.pt")
TEST_DATASET_PATH = os.path.join(SAVE_DIR, "test_dataset.pt")

# 加载 FRILL 模型
print("Loading FRILL model...")
module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/frill/1')


# 定义函数提取 FRILL 特征
def extract_frill_embeddings(audio_array, sampling_rate=16000):
    """
    使用 FRILL 模型从音频中提取嵌入特征。
    """
    # 确保音频是 16kHz
    if sampling_rate != 16000:
        raise ValueError("Sampling rate must be 16kHz for FRILL.")
    # 转换为浮点数并扩展维度
    audio_array = audio_array.astype(np.float32)
    batched_input = np.expand_dims(audio_array, axis=0)
    # 提取嵌入
    embeddings = module(batched_input)['embedding']
    return embeddings.numpy()

# 提取数据集的特征并保存
def preprocess_and_save_features(dataset, label2id, save_path, sampling_rate=16000):
    """
    提取数据集中的音频特征，并将特征和标签保存到文件。
    """
    features = []
    labels = []

    for sample in tqdm(dataset):
        # 加载音频数据
        audio = sample['audio']['array']
        label = sample['emotion']
        # 提取特征
        embedding = extract_frill_embeddings(audio, sampling_rate)
        features.append(embedding)
        labels.append(label2id[label])

    # 保存特征和标签
    torch.save({'features': features, 'labels': labels}, save_path)
    print(f"Features and labels saved to {save_path}.")


# 主函数
if __name__ == "__main__":
    # 加载保存的训练集和测试集
    train_dataset = torch.load(TRAIN_DATASET_PATH)
    test_dataset = torch.load(TEST_DATASET_PATH)

    # 提取情感标签
    emotions = sorted(set(train_dataset['emotion']))
    label2id = {value: key for key, value in enumerate(emotions)}
    id2label = {key: value for key, value in enumerate(emotions)}

    # 提取并保存训练集和测试集的特征
    TRAIN_FEATURES_PATH = os.path.join(SAVE_DIR, "train_features.pt")
    TEST_FEATURES_PATH = os.path.join(SAVE_DIR, "test_features.pt")

    if not os.path.exists(TRAIN_FEATURES_PATH):
        preprocess_and_save_features(train_dataset, label2id, TRAIN_FEATURES_PATH)
    if not os.path.exists(TEST_FEATURES_PATH):
        preprocess_and_save_features(test_dataset, label2id, TEST_FEATURES_PATH)