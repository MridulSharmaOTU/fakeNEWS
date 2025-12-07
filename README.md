# Fake News Detection with Classical Models and Pre-trained Transformers

Online misinformation spreads rapidly through social media and news sites, shaping public opinion on politics, health, and global events. Manual fact-checking cannot keep up with the volume of content, so automated fake news detection is a critical practical problem.

In this project, we use the **“Fake and Real News” dataset (ISOT Fake News detection dataset)** from Kaggle, which contains news articles labeled as *fake* or *real* along with their titles, full text, subject category, and publication date:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Our goal is to **build and compare different machine learning models** for automatically classifying news articles as fake or real based on their text content.

## 1. Dataset Description

The dataset is provided as two CSV files:
- `Fake.csv` – articles identified as fake news  
- `True.csv` – articles identified as real news  

Common columns include:
- **title** – headline of the article  
- **text** – main body of the article  
- **subject** – topic category (e.g., politics, world news)  
- **date** – publication date  

For this project, each **row is one news article**, and the **target label** will be:
- `1` = real news  
- `0` = fake news  

We will:
- Merge the two CSVs into a single dataset
- Add an explicit binary column `label` ∈ {0, 1}
- Use primarily the **text** (and optionally **title**) as input features

## 2. Learning Problem Formulation

### 2.1 Task Type

This is a **supervised binary text classification** problem:

- **Input**: A news article’s text (optionally concatenated with its title).
- **Output**: A binary label indicating whether the article is *fake* or *real*.

Formally, let:
- \( x_i \) be the text of article \( i \)
- \( y_i \in \{0, 1\} \) be the label, where 0 = fake, 1 = real

We want to learn a function
\[
f_\theta: \text{text} \rightarrow \{0, 1\}
\]
parameterized by \(\theta\), that predicts whether a previously unseen news article is fake or real.

### 2.2 Objective

We train \( f_\theta \) to minimize a standard **binary classification loss**, such as **binary cross-entropy**, on a labeled training set, and evaluate generalization on a held-out validation and test set.

Primary performance metrics:
- **Accuracy**
- **Precision, Recall, F1-score** for each class (fake vs real)
- **Confusion matrix** to see typical error types (e.g., fake classified as real)

## 3. Research Questions and Motivation

The high-level motivation is to understand **how much benefit we gain from modern pre-trained language models** compared to simpler approaches for fake news detection, under limited computational resources.

We focus on the following research questions:

1. **Modeling Power vs Simplicity**  
   How does a simple classical baseline (e.g., TF-IDF + linear classifier) compare to a small neural text classifier and a pre-trained Transformer (DistilBERT) on this fake vs real news task?

2. **Effect of Pre-training**  
   Does fine-tuning a pre-trained Transformer (DistilBERT) significantly improve performance over training a small neural network from scratch on the same dataset?

3. **Model Size vs Performance Trade-off**  
   How does **model size (number of parameters)** relate to the gains in performance? Is the extra complexity of a pre-trained Transformer justified by a meaningful improvement in accuracy/F1?

These questions align with a **comparative study of different machine learning approaches** on the same dataset, which is one of the recommended project types.

## 4. Planned Models and Experimental Setup

To answer the research questions, we will implement and compare the following models:

### 4.1 Baseline: Classical Machine Learning

- **Model**: TF-IDF features + simple linear classifier  
  (e.g., Logistic Regression or a one-hidden-layer feed-forward network)
- **Input representation**:  
  - Tokenize text
  - Compute TF-IDF vectors over the vocabulary
- **Purpose**:  
  - Provide a strong, cheap baseline
  - Show what can be achieved without deep learning or pre-training

### 4.2 Small Neural Network (from scratch)

- **Model**: Lightweight neural text classifier, e.g.:
  - An embedding layer + 1D CNN or BiLSTM over tokens, followed by a dense layer
- **Input representation**:
  - Learn word embeddings from scratch on this dataset
- **Purpose**:
  - Evaluate a “pure” neural network without external pre-training
  - Compare with both the classical baseline and the Transformer model

### 4.3 Pre-trained Transformer (fine-tuned)

- **Model**: DistilBERT (or similar small Transformer) fine-tuned for binary classification on this dataset :contentReference[oaicite:2]{index=2}
- **Input representation**:
  - Use the pre-trained DistilBERT tokenizer for subword tokens
- **Purpose**:
  - Leverage pre-training on large general corpora
  - Measure how much pre-training improves fake news detection vs. models trained only on this dataset

For each model we will record:
- Training and validation performance (Accuracy, F1)
- Test performance
- Approximate **number of trainable parameters**
- Basic runtime / training cost observations (e.g., epochs, training time per epoch)

## 5. Data Splitting, Evaluation, and Constraints

### 5.1 Data Splits

We will split the merged dataset into:
- **Training set** (e.g., 70%)
- **Validation set** (e.g., 15%)
- **Test set** (e.g., 15%)

The split will be **stratified** by the binary label to preserve the proportion of fake vs real news in each split.

### 5.2 Evaluation Protocol

1. Train each model on the training set.
2. Use the validation set to:
   - Tune hyperparameters (e.g., learning rate, max sequence length, regularization)
   - Select the best checkpoint for each model
3. Report final performance on the held-out test set:
   - Accuracy
   - Precision, Recall, F1 for fake and real classes
   - Confusion matrix

### 5.3 Computational Constraints

- We assume **limited compute** (likely CPU or a single modest GPU).
- To respect these constraints, we will:
  - Use relatively small models (e.g., DistilBERT instead of full BERT)
  - Limit maximum sequence length (truncate very long articles)
  - Use a modest number of epochs and early stopping
  - Optionally subsample the training data if needed for runtime

Model size (number of parameters) will be explicitly computed and reported for each model to relate performance gains to model complexity.

## 6. Expected Contributions

By the end of the project we expect to:

1. Provide a **clear comparative study** of:
   - A classical TF-IDF + linear model
   - A small neural network trained from scratch
   - A pre-trained Transformer fine-tuned on the fake news dataset

2. Quantify how much **pre-training and model size** contribute to performance on fake news detection, under realistic computational limits.

3. Discuss typical **failure modes**, such as:
   - Real news misclassified as fake (potential false positives)
   - Fake news misclassified as real (dangerous false negatives)

4. Reflect on how these results could inform **practical fake news detection systems**, which often must trade off accuracy, model size, and inference cost.