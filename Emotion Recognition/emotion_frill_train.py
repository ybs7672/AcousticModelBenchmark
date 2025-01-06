# 对frill提取的特征矩阵进行分类
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


# 数据集路径
SAVE_DIR = "/mnt/p-wangzixv/data_wav/emo_data/"
TRAIN_DATASET_PATH = os.path.join(SAVE_DIR, "train_dataset.pt")
TEST_DATASET_PATH = os.path.join(SAVE_DIR, "test_dataset.pt")
TRAIN_FEATURES_PATH = os.path.join(SAVE_DIR, "train_features.pt")
TEST_FEATURES_PATH = os.path.join(SAVE_DIR, "test_features.pt")


# 定义LSTM分类器
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, num_classes=6, num_layers=2, bidirectional=True):
        super(LSTMEmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x, lengths):
        # Packed sequence for handling variable-length inputs
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_x)
        # Unpack sequences
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Pooling over time dimension
        pooled_output = torch.mean(lstm_output, dim=1)  # 平均池化
        # Classification layer
        return self.fc(pooled_output)


# 自定义Dataset类
class AudioEmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.max_seq_len = max(f.shape[0] for f in features)  # 获取最长序列长度

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        seq_len = feature.shape[0]
        # Padding到max_seq_len
        padded_feature = np.zeros((self.max_seq_len, feature.shape[1]), dtype=np.float32)
        padded_feature[:seq_len, :] = feature
        return torch.tensor(padded_feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long), seq_len


def load_features_and_labels(file_path):
    """
    加载保存的音频特征和标签。
    """
    data = torch.load(file_path)
    features = data['features']  # List of (time, 2048) feature tensors
    labels = data['labels']  # List of numerical labels
    return features, labels


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels, lengths in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            features, labels, lengths = features.to(device), labels.to(device), lengths.cpu()
            optimizer.zero_grad()
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


# 测试函数
def evaluate_model(model, test_loader, device, id2label):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels, lengths in tqdm(test_loader, desc="Evaluating"):
            features, labels, lengths = features.to(device), labels.to(device), lengths.cpu()
            outputs = model(features, lengths)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 生成分类报告
    y_pred_labels = [id2label[pred] for pred in all_preds]
    y_true_labels = [id2label[label] for label in all_labels]
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=list(id2label.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载保存的训练集和测试集
    train_dataset = torch.load(TRAIN_DATASET_PATH)
    test_dataset = torch.load(TEST_DATASET_PATH)

    # 提取情感标签
    emotions = sorted(set(train_dataset['emotion']))
    label2id = {value: key for key, value in enumerate(emotions)}
    id2label = {key: value for key, value in enumerate(emotions)}

    # 加载特征和标签
    train_features, train_labels = load_features_and_labels(TRAIN_FEATURES_PATH)
    test_features, test_labels = load_features_and_labels(TEST_FEATURES_PATH)

    # 创建Dataset和DataLoader
    train_data = AudioEmotionDataset(train_features, train_labels)
    test_data = AudioEmotionDataset(test_features, test_labels)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=None)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=None)

    # 定义LSTM分类器
    model = LSTMEmotionClassifier(input_dim=2048, hidden_dim=128, num_classes=len(label2id), num_layers=2, bidirectional=True)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)

    # 测试模型
    evaluate_model(model, test_loader, device, id2label)