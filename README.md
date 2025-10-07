# Vietnamese hate speech detection
## 1. Introduce
The goal of this project is to classify sentiments in Vietnamese comments on social media to detect and prevent hate speech and offensive language. By doing so, the project contributes to creating a healthier and safer social media environment.

The project is implemented by training a model on the ViHSD dataset and using the TextCNN model to classify comments into three categories:

- **Clean**: Comments that do not contain any harassing behavior.

- **Offensive**: Comments that contain harassing content, including vulgar language, but are not directed at specific individuals.

- **Hate**: Comments that are hateful and offensive, directly targeting individuals or groups based on personal characteristics, religion, or nationality.

## 2. Dataset
The dataset used in this project is the ViHSD - Vietnamese Hate Speech Detection dataset.

ViHSD is a Vietnamese dataset collected from comments on popular social media platforms such as Facebook and YouTube. This dataset has been manually annotated to support research on the automatic detection of hate speech on social media platforms.

ViHSD contains 33,400 labeled comments, with a total vocabulary of 21,239 words. The comments in the dataset are divided into three categories based on the following labels: HATE (2), OFFENSIVE (1), and CLEAN (0).

```
@InProceedings{10.1007/978-3-030-79457-6_35,
author="Luu, Son T.
and Nguyen, Kiet Van
and Nguyen, Ngan Luu-Thuy",
editor="Fujita, Hamido
and Selamat, Ali
and Lin, Jerry Chun-Wei
and Ali, Moonis",
title="A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts",
booktitle="Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="415--426",
abstract="In recent years, Vietnam witnesses the mass development of social network users on different social platforms such as Facebook, Youtube, Instagram, and Tiktok. On social media, hate speech has become a critical problem for social network users. To solve this problem, we introduce the ViHSD - a human-annotated dataset for automatically detecting hate speech on the social network. This dataset contains over 30,000 comments, each comment in the dataset has one of three labels: CLEAN, OFFENSIVE, or HATE. Besides, we introduce the data creation process for annotating and evaluating the quality of the dataset. Finally, we evaluate the dataset by deep learning and transformer models.",
isbn="978-3-030-79457-6"
}
```
## 3. Method
### 3.1. Data preprocessing

The data preprocessing process was carried out through four main steps:

- **Text segmentation**: The text was segmented into individual words using the PyVi tool.

- **Stopword removal**: Words that do not carry significant meaning in a specific context, such as stopwords, were removed based on a predefined list in the **vietnamese-stopwords-dash.txt** file.

- **Emoji removal**: Emojis in the text were removed to ensure the data is suitable for subsequent processing steps.

- **Lowercasing**: All words in the text were converted to lowercase to ensure consistency.


  ### 3.2. Model
  <p align="center">
    <img src="https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/Transformer_Architecture.png" alt ="Transformer Architecture"> </p>
  <p align="center">
    <img src="https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/BERT_Architecture.png" alt ="BER Architecture"> </p>

The model used in this project is BERT. The configuration of the model is as follows:
- Base model: BERT-based-multilingual-cased
- Epochs: 4
- Batch size: 16
- Max sequence length: 100
- Optimizer: Adam
- Munual seed: 4

Architecture Overview
- Input layer: Tokenized Vietnamese comments, padded/truncated to 100 tokens.
- BERT Encoder: Multi-head self-attention layers capture bidirectional context.
- Dropout Layer: Reduces overfitting by randomly zeroing out some features during training.
- Dense Layer (Softmax): Outputs class probabilities across 3 categories.

## 4. Result
![](https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/clean.jpg)
![](https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/offensive.jpg)
![](https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/hate.jpg)


To better understand how the website operates, please watch this [video](https://github.com/chongan0224/CS221-Vietnamese-hate-speech-detection/blob/main/Vietnamese-hate-speech-detection-main/Vietnamese-hate-speech-detection-main/Image%20source/Results.mp4)

The BERT-muitilingual cased achieved an accuracy of 86.56% and an F1-macro score of 62.25% on the dataset.

However, the BERT-based model still encounters several challenges:

  - **Data Imbalance**: The model remains biased toward predicting the CLEAN label due to the large disparity in sample sizes among the three classes.

  - **Complex Informal Language**: Many Vietnamese social media comments include slang, abbreviations, and metaphorical expressions. Even though BERT captures context well, such informal and creative uses of language still pose difficulties for accurate classification.

  - **Limited Domain Adaptation**: The pre-trained multilingual BERT model was not specifically trained on Vietnamese social media text, so it may fail to fully capture cultural nuances and context-specific meanings.

To further enhance model performance, future improvements could include fine-tuning domain-specific language models such as PhoBERT, which is pre-trained on large-scale Vietnamese text data. Additionally, applying data balancing techniques or data augmentation can help mitigate class imbalance and improve robustness in detecting subtle or context-dependent hate speech.
