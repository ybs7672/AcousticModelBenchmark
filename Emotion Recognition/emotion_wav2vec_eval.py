from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import torch
import os
import pandas as pd
from emotion_wav2vec_train import process_RAVDESS, process_CREMA, process_TESS, process_SAVEE
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import Dataset
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader


# 保存训练集和测试集的路径
SAVE_DIR = "/mnt/p-wangzixv/data_wav/emo_data/"
TRAIN_DATASET_PATH = os.path.join(SAVE_DIR, "train_dataset.pt")
TEST_DATASET_PATH = os.path.join(SAVE_DIR, "test_dataset.pt")


def evaluate_model(model, test_dataloader, id2label):
    """
    在测试数据集上评估模型性能，并输出每种情感的准确率。

    Args:
        model: 训练好的模型。
        test_dataloader: 测试数据的 DataLoader。
        id2label: 字典，将类别 ID 映射为情感标签。

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 设置为评估模式
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_values, labels = batch  # 从测试数据集中获取输入和标签
            # 移动输入数据到模型所在的设备
            input_values = input_values.to(device)
            labels = labels.to(device)
            # 模型推理
            model_output = model(input_values)
            logits = model_output.logits  # 获取 logits
            predictions = torch.argmax(logits, dim=1)  # 获取预测类别
            # 收集预测和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 将预测值和真实值转换为对应的情感标签
    predicted_labels = [id2label[p] for p in all_predictions]
    true_labels = [id2label[t] for t in all_labels]

    # 计算并显示分类报告（包含每种情感的准确率、召回率和 F1-score）
    print("\nClassification Report:")
    report = classification_report(true_labels, predicted_labels, target_names=id2label.values(), digits=4)
    print(report)

    # 计算总准确率
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

    # 按情感显示准确率
    emotion_accuracies = {}
    for emotion in id2label.values():
        emotion_correct = sum((p == t) and (t == emotion) for p, t in zip(predicted_labels, true_labels))
        emotion_total = sum(t == emotion for t in true_labels)
        accuracy = emotion_correct / emotion_total if emotion_total > 0 else 0.0
        emotion_accuracies[emotion] = accuracy

    print("\nEmotion-wise Accuracy:")
    for emotion, accuracy in emotion_accuracies.items():
        print(f"{emotion}: {accuracy:.4f}")


if __name__ == "__main__":
    # 得到与训练相同的label2id和id2label
    ravdess_df = process_RAVDESS()
    crema_df = process_CREMA()
    tess_df = process_TESS()
    savee_df = process_SAVEE()
    df = pd.concat([
        ravdess_df,
        crema_df,
        tess_df,
        savee_df
    ], axis=0)
    df.drop('sex', axis=1, inplace=True)
    data = Dataset.from_pandas(df, preserve_index=False)
    dataset = data.cast_column("path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column('path', 'audio', )
    # emotions = set(dataset['emotion'])  # {'disgust', 'neutral', 'happy', 'sad', 'fear', 'angry'}
    emotions = sorted(set(dataset['emotion']))
    label2id = {value: key for key, value in enumerate(emotions)}
    id2label = {value: key for key, value in label2id.items()}

    # 加载划分好的测试集
    test_dataset = torch.load(TEST_DATASET_PATH)
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-100k-voxpopuli')
    def collate_batch(features):
        # Find the longest audio clip in the batch
        waveform = [sample['audio']['array'] for sample in features]
        labels = [label2id[sample['emotion']] for sample in features]
        # Pad all other audio clips to match this length
        waveform = feature_extractor(waveform, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        labels = torch.tensor(labels)

        return waveform, labels
    # 转换为 DataLoader（适用于 PyTorch 模型评估）
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

    MODEL_SAVE_PATH = "./emotion_recognition_model.pth"  # 替换为你的模型权重路径

    # 加载特征提取器和模型
    #feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-100k-voxpopuli')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'facebook/wav2vec2-base-100k-voxpopuli',
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label
    )
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 将模型移到 GPU 或 CPU

    evaluate_model(model, test_dataloader, id2label)
'''
if __name__ == "__main__":
    ravdess_df = process_RAVDESS()
    crema_df = process_CREMA()
    tess_df = process_TESS()
    savee_df = process_SAVEE()
    df = pd.concat([
        ravdess_df,
        crema_df,
        tess_df,
        savee_df
    ], axis=0)
    df.drop('sex', axis=1, inplace=True)
    data = Dataset.from_pandas(df, preserve_index=False)
    dataset = data.cast_column("path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column('path', 'audio', )
    #emotions = set(dataset['emotion'])  # {'disgust', 'neutral', 'happy', 'sad', 'fear', 'angry'}
    emotions = sorted(set(dataset['emotion']))
    label2id = {value: key for key, value in enumerate(emotions)}
    id2label = {value: key for key, value in label2id.items()}
    # 划分训练集、测试集
    def collate_batch(features):
        # Find the longest audio clip in the batch
        waveform = [sample['audio']['array'] for sample in features]
        labels = [label2id[sample['emotion']] for sample in features]
        # Pad all other audio clips to match this length
        waveform = feature_extractor(waveform, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        labels = torch.tensor(labels)

        return waveform, labels
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(dataset['test'], batch_size=16, shuffle=True, collate_fn=collate_batch)

    MODEL_SAVE_PATH = "./emotion_recognition_model.pth"  # 替换为你的模型权重路径

    # 加载特征提取器和模型
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-100k-voxpopuli')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'facebook/wav2vec2-base-100k-voxpopuli',
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label
    )
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 将模型移到 GPU 或 CPU

    evaluate_model(model, test_dataloader, id2label)
'''