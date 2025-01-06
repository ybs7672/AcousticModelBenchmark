import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification


#RAVDESS = r"E:\emo_data\ravdess-emotional-speech-audio\audio_speech_actors_01-24\\"
#CREMA = r"E:\emo_data\cremad\AudioWAV\\"
#TESS = r"E:\emo_data\toronto-emotional-speech-set-tess\tess toronto emotional speech set data\TESS Toronto emotional speech set data\\"
#SAVEE = r"E:\emo_data\surrey-audiovisual-expressed-emotion-savee\ALL\\"
RAVDESS = r"/mnt/p-wangzixv/data_wav/emo_data/audio_speech_actors_01-24/"
CREMA = r"/mnt/p-wangzixv/data_wav/emo_data/AudioWAV/"
TESS = r"/mnt/p-wangzixv/data_wav/emo_data/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
SAVEE = r"/mnt/p-wangzixv/data_wav/emo_data/ALL/"
# 保存训练集和测试集的路径
SAVE_DIR = "/mnt/p-wangzixv/data_wav/emo_data/"
TRAIN_DATASET_PATH = os.path.join(SAVE_DIR, "train_dataset.pt")
TEST_DATASET_PATH = os.path.join(SAVE_DIR, "test_dataset.pt")


def process_RAVDESS():
    ravdess_dir_lis = os.listdir(RAVDESS)
    path_list = []
    gender_list = []
    emotion_list = []
    emotion_dic = {
        '03': 'happy',
        '01': 'neutral',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
    }
    for directory in ravdess_dir_lis:
        actor_files = os.listdir(os.path.join(RAVDESS, directory))
        for audio_file in actor_files:
            part = audio_file.split('.')[0]
            key = part.split('-')[2]
            if key in emotion_dic:
                gender_code = int(part.split('-')[6])
                path_list.append(f"{RAVDESS}{directory}/{audio_file}")
                gender_list.append('female' if gender_code & 1 == 0 else 'male')
                emotion_list.append(emotion_dic[key])
    ravdess_df = pd.concat([
        pd.DataFrame(path_list, columns=['path']),
        pd.DataFrame(gender_list, columns=['sex']),
        pd.DataFrame(emotion_list, columns=['emotion'])
    ], axis=1)
    return ravdess_df

def process_CREMA():
    crema_dir_list = os.listdir(CREMA)
    path_list = []
    gender_list = []
    emotion_list = []
    emotion_dic = {
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fear',
        'DIS': 'disgust',
    }
    female_id_list = [
        '1002', '1003', '1004', '1006', '1007', '1008', '1009', '1010', '1012', '1013', '1018',
        '1020', '1021', '1024', '1025', '1028', '1029', '1030', '1037', '1043', '1046', '1047',
        '1049', '1052', '1053', '1054', '1055', '1056', '1058', '1060', '1061', '1063', '1072',
        '1073', '1074', '1075', '1076', '1078', '1079', '1082', '1084', '1089', '1091',
    ]
    for audio_file in crema_dir_list:
        part = audio_file.split('_')
        key = part[2]
        if key in emotion_dic and part[3] == 'HI.wav':
            path_list.append(f"{CREMA}{audio_file}")
            gender_list.append('female' if part[0] in female_id_list else 'male')
            emotion_list.append(emotion_dic[key])
    crema_df = pd.concat([
        pd.DataFrame(path_list, columns=['path']),
        pd.DataFrame(gender_list, columns=['sex']),
        pd.DataFrame(emotion_list, columns=['emotion'])
    ], axis=1)
    return crema_df

def process_TESS():
    tess_dir_list = os.listdir(TESS)
    path_list = []
    gender_list = []
    emotion_list = []
    emotion_dic = {
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'Sad': 'sad',
        'angry': 'angry',
        'fear': 'fear',
        'disgust': 'disgust',
    }
    for directory in tess_dir_list:
        audio_files = os.listdir(os.path.join(TESS, directory))
        for audio_file in audio_files:
            part = audio_file.split('.')[0]
            key = part.split('_')[2]
            if key in emotion_dic:
                path_list.append(f"{TESS}{directory}/{audio_file}")
                gender_list.append('female')  # female only dataset
                emotion_list.append(emotion_dic[key])
    tess_df = pd.concat([
        pd.DataFrame(path_list, columns=['path']),
        pd.DataFrame(gender_list, columns=['sex']),
        pd.DataFrame(emotion_list, columns=['emotion'])
    ], axis=1)
    return tess_df

def process_SAVEE():
    savee_dir_list = os.listdir(SAVEE)
    path_list = []
    gender_list = []
    emotion_list = []
    emotion_dic = {
        'h': 'happy',
        'n': 'neutral',
        'sa': 'sad',
        'a': 'angry',
        'f': 'fear',
        'd': 'disgust'
    }
    for audio_file in savee_dir_list:
        part = audio_file.split('_')[1]
        key = part[:-6]
        if key in emotion_dic:
            path_list.append(f"{SAVEE}{audio_file}")
            gender_list.append('male')  # male only dataset
            emotion_list.append(emotion_dic[key])
    savee_df = pd.concat([
        pd.DataFrame(path_list, columns=['path']),
        pd.DataFrame(gender_list, columns=['sex']),
        pd.DataFrame(emotion_list, columns=['emotion'])
    ], axis=1)
    return savee_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def collate_batch(features):
        # Find the longest audio clip in the batch
        waveform = [sample['audio']['array'] for sample in features]
        labels = [label2id[sample['emotion']] for sample in features]
        # Pad all other audio clips to match this length
        waveform = feature_extractor(waveform, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        labels = torch.tensor(labels)

        return waveform, labels
    # 划分训练集、测试集
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(dataset['test'], batch_size=16, shuffle=True, collate_fn=collate_batch)
    # 保存划分的训练集、测试集
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    torch.save(train_dataset, TRAIN_DATASET_PATH)
    torch.save(test_dataset, TEST_DATASET_PATH)
    print(f"Datasets saved to {SAVE_DIR}")

    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-100k-voxpopuli')
    model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base-100k-voxpopuli',
                                                              num_labels=len(label2id),
                                                              label2id=label2id,
                                                              id2label=id2label
                                                              )
    model = model.to(device)

    # 冻结主干层
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    # 优化器只优化分类头
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=5e-4)

    num_epochs = 50
    criterion = torch.nn.CrossEntropyLoss()
    # 训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            input_values, labels = batch
            input_values = input_values.to(device)
            labels = labels.to(device)
            # Forward pass
            model_output = model(input_values, labels=labels)
            loss = model_output.loss
            # Compute loss - target_lengths must match the number of labels in the batch
            total_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

    # 保存模型
    MODEL_SAVE_PATH = "./emotion_recognition_model.pth"  # 保存路径
    # 确保路径存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # 保存模型权重
    if model:  # 检查 model 是否存在
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        print("Model not found. Please ensure the model is trained and exists in memory.")