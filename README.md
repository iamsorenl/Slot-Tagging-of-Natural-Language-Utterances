# Slot Tagging of Natural Language Utterances Using BiLSTM with Attention

This project implements a **BiLSTM with Attention** model for slot tagging of natural language utterances using **PyTorch**. The model uses **GloVe embeddings** for word representation and is evaluated using **sklearn** and **seqeval** metrics for token and sequence-based F1 scores.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

Slot tagging is a key component of Natural Language Understanding (NLU) systems, involving the labeling of input utterances with slot tags in the **Inside-Outside-Beginning (IOB)** format. This project aims to solve the problem using a **BiLSTM model with attention**, leveraging **GloVe embeddings** for enriched word representations.

The dataset consists of natural language utterances paired with IOB slot tags. The system:

- Uses **BiLSTM with Attention** to capture contextual dependencies.
- Employs **GloVe embeddings** to enhance semantic understanding.
- Evaluates performance using **sklearn** for token-level F1 scores and **seqeval** for sequence-based F1 scores.

---

## Requirements

To set up the project, ensure you have the following dependencies installed:

- Python >= 3.11
- PyTorch
- scikit-learn
- seqeval
- gensim
- pandas
- numpy

### Installing Dependencies

You can install all necessary packages by running:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/slot-tagging-bilstm.git
cd slot-tagging-bilstm
```

### 2. Create a Virtual Environment

To avoid package conflicts, it is recommended to use a virtual environment:

```bash
python3 -m venv venv
# Activate on MacOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```

### 3. Download Pre-trained GloVe Embeddings

The project uses **GloVe embeddings** from `glove-wiki-gigaword-100`. You can download them using:

```bash
python -m spacy download en_core_web_md
```

---

## Dataset Structure

Ensure the following files are present in the project directory:

- `hw2_train.csv`: Training dataset with utterances and slot tags.
- `hw2_test.csv`: Test dataset containing utterances.
- `sampleSubmission.csv`: Example output format for predictions.

### Example Data Format

**Training Dataset (hw2_train.csv):**

| ID  | UTTERANCE                        | IOB Slot Tags                         |
| --- | -------------------------------- | ------------------------------------- |
| 1   | Show me movies directed by Nolan | O O B-movie O O B-director I-director |

**Test Dataset (hw2_test.csv):**

| ID  | UTTERANCE                        |
| --- | -------------------------------- |
| 1   | Show me movies directed by Nolan |

---

## Usage

### Running the Training and Prediction Pipeline

To train the model and generate predictions:

```bash
python run.py hw2_train.csv hw2_test.csv output.csv
```

This command will:

1. Train the BiLSTM model with attention.
2. Process the test data to predict slot tags.
3. Save predictions in `output.csv`.

### Output Format

The output file `output.csv` will be structured as:

| ID  | IOB Slot Tags                         |
| --- | ------------------------------------- |
| 1   | O O B-movie O O B-director I-director |

---

## Training the Model

The model is trained using a **BiLSTM with attention** architecture, integrating **GloVe embeddings** for semantic richness. Key features:

- **Embedding Layer**: Pre-trained GloVe embeddings enrich token representations.
- **BiLSTM Layer**: Processes tokens in both forward and backward directions for comprehensive context capture.
- **Attention Mechanism**: Emphasizes key tokens in each sequence.

### Hyperparameters

- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Hidden Dimension**: 128
- **Number of LSTM Layers**: 1
- **Dropout**: None (for final model configuration)

---

## Evaluation

### Metrics

The model's performance is evaluated using:

- **Token-level F1 Score (sklearn)**
- **Sequence-based F1 Score (seqeval)**

### Example Results

**Sample Evaluation Metrics:**

| Metric     | Value  |
| ---------- | ------ |
| Accuracy   | 75.37% |
| Sklearn F1 | 0.950  |
| Seqeval F1 | 0.839  |

---

## Hyperparameter Tuning

Key hyperparameters such as learning rate, hidden dimensions, and dropout were tuned to optimize performance. The inclusion of **GloVe embeddings** and the use of **attention mechanisms** were found to significantly enhance performance.

---

## Limitations and Future Work

- **Dataset Size**: Limited training data may affect generalization.
- **Hyperparameter Search**: Current tuning is manual; automated methods may yield better results.
- **Exploration of Transformers**: Consideration for using Transformer-based architectures such as BERT.
