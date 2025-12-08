# Fake News Detection – Worksheet README

This worksheet demonstrates how to fine‑tune a pre‑trained neural network to detect fake news using a real‑world dataset from Kaggle.

## 1. Problem Overview

Online misinformation spreads quickly through social media and news sites, influencing public opinion about politics, health, and current events. Manually checking every article is impossible, so we treat **fake news detection** as a supervised machine learning problem:

> **Goal:** Given the text of a news article, automatically predict whether it is **fake** or **real**.

In this worksheet we:

* Load and clean a labeled fake/real news dataset.
* Split the data into training, validation, and test sets.
* Fine‑tune a pre‑trained **DistilBERT** model (a small Transformer) for binary classification.
* Evaluate the model using accuracy, precision, recall, F1‑score, and a confusion matrix.

## 2. Dataset

We use the Kaggle **“Fake and Real News”** dataset (ISOT Fake News Detection dataset), which is provided in two CSV files:

* `Fake.csv` – articles labeled as fake news
* `True.csv` – articles labeled as real news

Important columns:

* `title` – headline of the article
* `text` – main body of the article
* `subject` – topic category (e.g., politics)
* `date` – publication date

For this worksheet:

* Each **row** corresponds to **one news article**.
* We create a binary **label** column:

  * `0` = fake
  * `1` = real
* We combine `title` and `text` into a single input field used by the model.

## 3. Learning Problem

This is a **supervised binary text classification** task.

* **Input:** `x` = article text (title + body).
* **Target:** `y ∈ {0, 1}` where 0 = fake, 1 = real.

We learn a function

[
f_θ(x) → {0, 1}
]

parameterized by (θ), such that (f_θ(x)) predicts the correct label for new, unseen articles.

We train the model to minimize a **cross‑entropy loss** over the labeled training set and monitor validation metrics to detect overfitting.

## 4. Data Splits and Evaluation

We split the merged dataset into three parts:

* **Training set:** 70% of the data
* **Validation set:** 15% of the data
* **Test set:** 15% of the data

The splits are **stratified** by the label so that each split keeps a similar proportion of fake and real articles.

We report:

* **Accuracy**
* **Precision, Recall, F1‑score** for both classes
* **Confusion matrix** (fake vs real)

These metrics are computed on the held‑out **test set** after training.

## 5. Neural Network Architecture

The main model used in this worksheet is a **pre‑trained Transformer**:

* **Backbone:** DistilBERT (a smaller, faster version of BERT)
* **Task head:** A small classification head on top of DistilBERT

Architecture summary:

1. **Tokenizer:**

   * Converts raw text into subword token IDs and attention masks.
   * Truncates or pads sequences to a fixed maximum length.
2. **DistilBERT encoder:**

   * Multiple Transformer layers with self‑attention and feed‑forward blocks.
   * Produces contextual embeddings for each token in the sequence.
3. **Classification head:**

   * Uses the pooled representation (based on the first token) as a summary of the article.
   * Applies a small feed‑forward network and outputs logits for two classes: fake vs real.
4. **Loss:**

   * Cross‑entropy loss between predicted logits and true labels.

We fine‑tune **all** parameters of the model (both the encoder and the classification head) on the fake vs real labels.

## 6. Implementation Details

The worksheet is implemented in **PyTorch** with **Lightning** to structure the training loop. Key steps in the notebook:

1. **Setup and configuration**
   Import libraries, detect the GPU (RTX 4070), and set global hyperparameters (batch size, max sequence length, etc.).

2. **Data loading and preprocessing**
   Load `Fake.csv` and `True.csv`, create the `label` column, combine title and text, and split the data into train/validation/test sets.

3. **Tokenization and Dataset class**
   Initialize the DistilBERT tokenizer and define a custom `NewsDataset` to convert text into token IDs, attention masks, and labels.

4. **DataLoaders**
   Create `DataLoader`s for training, validation, and testing so that data can be processed in mini‑batches.

5. **Model definition**
   Define a `DistilBertClassifier` LightningModule that wraps the pre‑trained DistilBERT model, classification head, loss, and accuracy metric.

6. **Training and testing**
   Use `Trainer.fit` to train the model on the training set while monitoring validation loss and accuracy, then `Trainer.test` to evaluate on the test set.

7. **Metrics and plots**
   Load `metrics.csv` from `lightning_logs/version_0`, plot validation accuracy over epochs, and print the classification report and confusion matrix.

### Run the GUI from source

1. Place the exported model artifacts (including `model.safetensors`) in `export/fake_news_model/`.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Launch the app: `python export/gui_app.py`.

### Build a standalone executable with PyInstaller

1. Install build dependencies (PyInstaller is listed in `requirements.txt`):
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Create a one-file bundle that includes the model assets:
   ```bash
   pyinstaller --onefile --name fake-news-detector --add-data "export/fake_news_model:fake_news_model" export/gui_app.py
   ```
   On Windows, replace the colon in `--add-data` with a semicolon: `export/fake_news_model;fake_news_model`.
3. Run the executable from `dist/` (e.g., `./dist/fake-news-detector` on Linux/macOS or `dist\\fake-news-detector.exe` on Windows). The GUI will load the bundled model files automatically.