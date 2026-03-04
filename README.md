# NLP PCL Classification Coursework

This repository contains the implementation for a Natural Language Processing coursework project on **detecting Patronizing and Condescending Language (PCL)** using the *Don't Patronize Me!* dataset.

The project includes:
- Exploratory Data Analysis (EDA)
- A baseline model
- An improved model based on **class-weighted fine-tuning and threshold tuning of RoBERTa**

The task is **binary classification**, where the goal is to predict whether a paragraph contains PCL.

```
0 = No PCL
1 = PCL
```

---

# Repository Structure

```
NLP_PCL_Coursework/
│
├── data/
│   └── raw/
│       ├── dontpatronizeme_pcl.tsv
│       ├── train_semeval_parids-labels.csv
│       ├── dev_semeval_parids-labels.csv
│       └── task4_test.tsv
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   └── 02_Baseline.ipynb     # Baseline RoBERTa model
│
├── BestModel/
│   ├── 03_BestModel.ipynb    # Final improved model
│   ├── model/                # Saved trained model and tokenizer
│   ├── dev.txt               # Predictions for the official dev set
│   ├── test.txt              # Predictions for the official test set
│   └── model_metadata.json   # Model configuration and threshold
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

# Dataset

This coursework uses the **Don't Patronize Me! dataset**, which contains paragraphs extracted from news articles referencing vulnerable communities such as refugees, homeless individuals, migrants, and poor families.

The dataset includes:

| File | Description |
|-----|-------------|
| `dontpatronizeme_pcl.tsv` | Main dataset containing paragraphs |
| `train_semeval_parids-labels.csv` | Training labels |
| `dev_semeval_parids-labels.csv` | Development set labels |
| `task4_test.tsv` | Official test set (labels hidden) |

The goal is to train a model that detects **patronizing or condescending language** in these texts.

---

# Environment Setup

All required Python dependencies are listed in `requirements.txt`.

To install them:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- `transformers` for pretrained language models
- `torch` for deep learning
- `scikit-learn` for evaluation metrics
- `pandas` and `numpy` for data processing
- `matplotlib` for visualisations

---

# Exploratory Data Analysis

EDA is implemented in:

```
notebooks/01_EDA.ipynb
```

Two main analysis techniques were applied:

### 1. Statistical Profiling
- Class distribution
- Token length distribution
- Identification of sequence length requirements

### 2. Lexical Analysis
- N-gram frequency analysis
- Identification of words and phrases more common in PCL examples

These analyses informed preprocessing decisions such as maximum sequence length and highlighted the strong **class imbalance** present in the dataset.

---

# Baseline Model

The baseline model is implemented in:

```
notebooks/02_Baseline.ipynb
```

The baseline uses:

- **RoBERTa-base**
- Standard fine-tuning for binary classification
- Default probability threshold of **0.5**

This serves as a reference point for evaluating improved approaches.

---

# Proposed Approach (BestModel)

The final model and training pipeline are implemented in:

```
BestModel/03_BestModel.ipynb
```

This approach improves upon the baseline using two modifications.

### Class-weighted loss

Because the dataset is highly imbalanced (~90% No PCL, ~10% PCL), a **class-weighted cross-entropy loss** is used during training to increase the penalty for misclassifying PCL examples.

### Threshold tuning

Instead of using the default classification threshold of **0.5**, the decision threshold is tuned on the development set to maximise the **F1 score of the positive class (PCL)**.

This produces a better balance between precision and recall.

---

# Model Outputs

The required prediction files are stored in:

```
BestModel/
```

| File | Description |
|-----|-------------|
| `dev.txt` | Predictions for the official development set |
| `test.txt` | Predictions for the official test set |

Each file contains **one prediction per line**:

```
0
1
0
0
1
...
```

Where:

```
0 = No PCL
1 = PCL
```

The number of lines in each file matches the number of examples in the corresponding dataset.

---

# Saved Model

The trained model and tokenizer are saved in:

```
BestModel/model/
```

This allows the model to be reloaded and used for inference without retraining.

Example:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("BestModel/model")
tokenizer = AutoTokenizer.from_pretrained("BestModel/model")
```

---

# Reproducibility

To reproduce the results:

1. Install dependencies from `requirements.txt`
2. Run this notebook:

```
BestModel/03_BestModel.ipynb
```

3. The final predictions will be generated in:

```
BestModel/dev.txt
BestModel/test.txt
```
