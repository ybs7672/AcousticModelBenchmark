# Emotion Recognition

## About
This is a demo for audio emotion classification. First, a pre-trained wav2vec network (`wav2vec2-base-100k-voxpopuli` or `frill`) is used to extract features from the audio. Then, an LSTM network is employed to classify the audio features. We utilized audio data from four datasets—CREMA-D, RAVDESS, TESS, and SAVEE and split them into training and testing sets in an 8:2 ratio. Our classification targets include six emotions: happy, neutral, sad, angry, fear, and disgust.

## Results
On the test set obtained from the split, the audio emotion classification achieved an accuracy of **0.35** when using the wav2vec2-base-100k-voxpopuli network for feature extraction, and an accuracy of **0.91** when using the frill network for feature extraction.

## Procedure
1. Download CREMA-D, RAVDESS, TESS, and SAVEE dataset from [Kaggle](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition/input)
2. Wav2vec2 feature extract

    1. `emotion_wav2vec_train.py` to organize the dataset, split it into training and testing sets, and then proceed with audio classification training
      ```
      python emotion_wav2vec_train.py
      ```

    2. `emotion_wav2vec_eval.py` to evaluate the classification accuracy of the trained model on the test set
      ```
      python emotion_wav2vec_eval.py
      ```

3. FRILL feature extract

    1. `emotion_frill_feature.py` to use FRILL to extract the audio features of the pre-split training set from `emotion_wav2vec_train.py`
      ```
      python emotion_frill_feature.py
      ```

    2. `emotion_frill_train.py` to train and test an LSTM audio feature classification network
      ```
      python emotion_frill_train.py
      ```
