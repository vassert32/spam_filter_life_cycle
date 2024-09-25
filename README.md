# Spam Filter: Comparing Vectorization Methods (TF-IDF, Word2Vec, BERT)

This project focuses on building and evaluating a spam classification model using various machine learning algorithms and text vectorization techniques, such as **TF-IDF**, **Word2Vec**, and **BERT**. It demonstrates how natural language processing (NLP) techniques can be used to classify emails as spam or ham.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Text Vectorization Methods](#text-vectorization-methods)
  - [TF-IDF](#tf-idf)
  - [Word2Vec](#word2vec)
  - [BERT](#bert)
- [Classification Algorithms](#classification-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Word Clouds](#word-clouds)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to build a spam filter using machine learning classification methods and compare different text vectorization techniques: TF-IDF, Word2Vec, and BERT. The project includes a detailed performance analysis of these methods in terms of speed and classification quality.

The project aims to classify emails into two categories: **spam** and **ham**. This is achieved through:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- NLP Preprocessing
- Model Building with machine learning classifiers
- Model Evaluation

## Technologies Used

- **Python** for data preprocessing, vectorization, and model training
- **scikit-learn** for TF-IDF vectorization and classification models
- **Gensim** for Word2Vec vectorization
- **Transformers** from Hugging Face for BERT embeddings
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for visualizing results

## Dataset

We use a publicly available dataset of spam and non-spam messages (SMS). Each message is labeled as either "spam" or "ham" (not spam). The dataset has been split into training and test sets to evaluate model performance.

## Text Vectorization Methods

### TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect the importance of a word in a document relative to the entire dataset. It is a widely used method for transforming raw text data into numerical vectors suitable for machine learning models.

### Word2Vec

asjdllkaskldjaklsdkl

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that generates contextual embeddings, providing deeper and richer word representations based on the entire sentence context, not just isolated words.

## Classification Algorithms

The following classification algorithms were applied using the vectors generated by each method:

- **Naive Bayes**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **k-Nearest Neighbors (k-NN)**

## Evaluation Metrics

To compare the effectiveness of the vectorization techniques, the models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Training and inference time**

## Results

The following summarizes the performance comparison across the vectorization methods (!remark! -- max values of classifiers):

| Vectorization | Accuracy | Precision | Recall | F1-Score | Training Time |
|---------------|----------|-----------|--------|----------|---------------|
| **TF-IDF**    | 0.97     | 1         | 0.83   | 0.90     | Fast          |
| **W2V**       | 0.96     | 0.91      | 0.84   | 0.85     | Medium        |
| **BERT**      | 0.98     | 0.98      | 0.92   | 0.93     | Slow          |

The detailed analysis shows that while BERT provides the highest accuracy and F1-score, it also requires significantly more computational resources and time compared to TF-IDF and W2V.

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/vasser32/spam-filter-life-cycle.git
```
2. Install the required dependencies
```
pip install -r requirements.txt
```

## Conclusion

This project demonstrated the effectiveness of various machine learning models and text vectorization techniques in classifying emails as spam or ham. By applying methods such as TF-IDF, Word2Vec, and BERT, we explored different approaches to feature extraction and text representation. 

The results showed that **Random Forest** and **SVM** classifiers performed consistently well across different vectorization techniques, particularly with TF-IDF and BERT. **Naive Bayes**, while effective with TF-IDF, struggled with more complex vectorizations like Word2Vec and BERT. The **K-Nearest Neighbors** classifier had difficulties in handling class imbalances, leading to lower performance.

Overall, the combination of robust preprocessing, careful feature extraction, and well-suited classifiers proved to be essential in building a reliable spam filter. For future improvements, additional techniques like data augmentation or deeper fine-tuning of BERT could further enhance model accuracy, especially when dealing with more complex datasets or heavily imbalanced data.
