# Sentiment Analysis on Tweets Using LSTM in PyTorch

This project performs sentiment analysis on tweets from the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The goal is to classify tweets into positive and negative sentiments using an LSTM model implemented in PyTorch. The notebook covers the entire pipeline—from data preprocessing and cleaning to model training, and evaluation using metrics such as ROC AUC, recall, precision, and F1 score.

---

## Table of Contents

- [Sentiment Analysis on Tweets Using LSTM in PyTorch](#sentiment-analysis-on-tweets-using-lstm-in-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset Overview](#dataset-overview)
  - [Data Preparation](#data-preparation)
    - [Text Cleaning and Preprocessing](#text-cleaning-and-preprocessing)
    - [Tokenization, Encoding, and Padding](#tokenization-encoding-and-padding)
  - [Modeling](#modeling)
    - [LSTM Model Architecture](#lstm-model-architecture)
    - [Hyperparameter Choices and Rationale](#hyperparameter-choices-and-rationale)
  - [Model Training](#model-training)
    - [Optimizer, Loss Function, and Learning Rate](#optimizer-loss-function-and-learning-rate)
  - [Evaluation and Analysis](#evaluation-and-analysis)
    - [Metrics: ROC AUC, Recall, Precision, F1 Score](#metrics-roc-auc-recall-precision-f1-score)
    - [Confusion Matrix and ROC Curve Visualization](#confusion-matrix-and-roc-curve-visualization)
  - [Usage](#usage)

---

## Project Overview

In this project, we build a sentiment classifier for tweets using the Sentiment140 dataset. The model is designed to predict sentiment polarity (positive or negative) for each tweet. Key highlights include:

- **Text Preprocessing:** Cleaning and preparing tweet data.
- **LSTM Modeling:** Building an LSTM network in PyTorch to capture sequential patterns in text.
- **Evaluation:** Using multiple metrics to assess model performance, with visualizations for deeper analysis.

---

## Dataset Overview

**Sentiment140 Dataset**

- **Source:** [Sentiment140](http://help.sentiment140.com/for-students/)
- **Size:** Approximately 1.6 million tweets with sentiment polarity labels.
- **Labels:** Originally encoded as `0` (negative) and `4` (positive); mapped to binary labels (`0` and `1`) for this project.
- **Content:** Includes tweet text and additional metadata, though only the tweet text is used for this analysis.

This dataset is a widely accepted benchmark for sentiment analysis tasks, especially in social media contexts.

---

## Data Preparation

### Text Cleaning and Preprocessing

1. **Lowercasing:**  
   All tweet text is converted to lowercase to ensure uniformity.

2. **Removing Mentions, URLs, and Hashtags:**  
   Regular expressions (regex) are used to remove patterns such as `@username`, URLs (e.g., `http://...`), and the `#` symbol from hashtags.

3. **Punctuation Removal:**  
   Punctuation is stripped out to focus on the core words.

4. **Stopwords Removal:**  
   The NLTK stopwords list is used to filter out common words (e.g., "the", "is", "and") that add little sentiment value.

### Tokenization, Encoding, and Padding

- **Tokenization:**  
  Cleaned text is split into individual tokens using a simple whitespace tokenizer.

- **Vocabulary Building and Encoding:**  
  A vocabulary dictionary is built from the dataset, mapping each token to a unique integer index. Special tokens like `<pad>` (for padding) and `<unk>` (for unknown words) are added.

- **Padding:**  
  Tweets are padded to a fixed sequence length—determined based on data distribution (for example, the 95th percentile of tweet lengths)—to facilitate efficient batch processing during training.

---

## Modeling

### LSTM Model Architecture

The LSTM model is implemented in PyTorch and includes the following layers:

1. **Embedding Layer:**  
   Converts token indices into dense embedding vectors, capturing semantic relationships between words.

2. **Dropout Layer:**  
   A dropout layer (with a dropout rate of `0.5`) is applied **before** the LSTM layer to prevent overfitting.

3. **Single LSTM Layer:**  
   Processes the sequence of embeddings to capture temporal dependencies in text. The model uses a **single LSTM layer** with `hidden_dim = 300`.

4. **Fully Connected Layer:**  
   Maps the final hidden state of the LSTM to a binary output (representing positive or negative sentiment).

4. **Sigmoid Layer**  
   Maps output of FC layer to values between 0 to 1, which represents probability.


### Hyperparameter Choices and Rationale

- **`vocab_size`:**  
  The size of the vocabulary built from the dataset (e.g., ~447,915 tokens).

- **`embedding_dim = 300`:**  
  Chosen to provide a meaningful representation of words while keeping the model computationally feasible.

- **`hidden_dim = 300`:**  
  Provides sufficient capacity for the LSTM to model patterns without overfitting.

- **`output_dim`:**  
  A single output unit is used for binary classification.

- **`n_layers = 1`:**  
  A single LSTM layer is used to maintain simplicity while capturing sequential dependencies.

- **`dropout = 0.5`:**  
  Applied before the LSTM layer to prevent overfitting by randomly zeroing out some activations during training. This helps balance training and validation losses.

---

## Model Training

### Optimizer, Loss Function, and Learning Rate

- **Loss Function:**  
  Binary Cross Entropy Loss (BCELoss) is used because it is suitable for binary classification tasks.

- **Optimizer:**  
  The Adam optimizer is chosen for its adaptive learning rate, which often leads to faster convergence.

- **Learning Rate:**  
  The model is trained with an initial learning rate of `0.001`.

- **Loss Tracking:**  
  The average loss is calculated separately for training and validation at each epoch. Dropout (`0.5`) was used to help maintain a balance between these two losses and mitigate overfitting.

---

## Evaluation and Analysis

### Metrics: ROC AUC, Recall, Precision, F1 Score

After training, the model is evaluated using several metrics:

- **ROC AUC:**  
  Measures the ability of the model to distinguish between classes across various thresholds.
  
- **Recall:**  
  The ratio of true positives to the sum of true positives and false negatives.

- **Precision:**  
  The ratio of true positives to the sum of true positives and false positives.

- **F1 Score:**  
  The harmonic mean of precision and recall, balancing both metrics.

### Confusion Matrix and ROC Curve Visualization

- **Confusion Matrix:**  
  A heatmap visualization of the confusion matrix displays the counts of true positives, true negatives, false positives, and false negatives, providing insight into model performance.

- **ROC Curve:**  
  The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR).

---

## Usage

**Install dependencies:**
   
```bash
   pip install torch scikit-learn nltk seaborn matplotlib
```
---

