# Spam Filter: Comparing Vectorization Methods (TF-IDF, Word2Vec, BERT)

## Project Overview

The goal of this project is to build a spam filter using machine learning classification methods and compare different text vectorization techniques: TF-IDF, Word2Vec, and BERT. The project includes a detailed performance analysis of these methods in terms of speed and classification quality.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Text Vectorization Methods](#text-vectorization-methods)
  - [TF-IDF](#tf-idf)
  - [GloVe](#glove)
  - [BERT](#bert)
- [Classification Algorithms](#classification-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

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

###TF-IDF

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

The following summarizes the performance comparison across the vectorization methods:

| Vectorization | Accuracy | Precision | Recall | F1-Score | Training Time |
|---------------|----------|-----------|--------|----------|---------------|
| **TF-IDF**    | *        | *         | *      | *        | Fast          |
| **W2V**       | *        | *         | *      | *        | Medium        |
| **BERT**      | *        | *         | *      | *        | Slow          |

The detailed analysis shows that while BERT provides the highest accuracy and F1-score, it also requires significantly more computational resources and time compared to TF-IDF and GloVe.

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/vasser32/spam-filter-life-cycle.git
```
2. Install the required dependencies
```
pip install -r requirements.txt
```
3. 
