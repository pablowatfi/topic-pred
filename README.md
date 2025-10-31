# topic-pred

A minimal Python starter project for topic prediction (placeholder implementation).

What is included
- `requirements.txt` – runtime dependencies to install (minimal)
- `src/topic_pred` – package with a `predict_template` function

## How to create the artifacts
- You need to download the data in the 'data' folder (find them in https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations.). 3 files are needed:
  - `content.csv`
  - `correlations.csv`
  - `topics.csv`
- Then run in full the notebook in `notebooks/topic_prediction_training_refactored.ipynb`

# Solution

Content-to-Topic prediction system for K-12 educational materials using semantic embeddings.

## Overview of Approach

This solution uses a **semantic embedding similarity approach** to match content to topics:

1. **Text Preprocessing**: Content titles and descriptions are concatenated and cleaned
2. **Embedding Generation**: Uses `paraphrase-multilingual-MiniLM-L12-v2` sentence transformer to encode both content and topics into dense vector representations
3. **Similarity Matching**: Computes cosine similarity between content and topic embeddings
4. **Filtering & Ranking**: Returns top-k topics (default k=3) with similarity scores above threshold (default 0.3)

**Rationale**: Embedding-based approaches capture semantic meaning effectively and generalize well to new content without requiring extensive training data. The multilingual model ensures robustness across varied educational content.

## Code Access Point

**Start here**: `topic_prediction_training_refactored.ipynb`
- Notebook contains full pipeline: data loading, preprocessing, model training, and artifact generation
- Run cells sequentially from top to bottom to regenerate artifacts

**For predictions**:
- `predict_template.py` → `TopicPredictor` class
- Uses pre-generated artifacts in `artifacts/topic_predictor_direct_model.pkl`

**Supporting utilities**:
- `preprocess_methods.py`: Text preprocessing and embedding generation
- `predict_method.py`: Core prediction logic using cosine similarity
- `io.py`: Data loading helpers
- `defaults.py`: Configuration constants

## Metrics

| Metric | Score |
|--------|-------|
| Recall@3 | 0.4576 |
| Precision@3 | 0.1876 |
| F2-Score@3 | 0.3587 |

*Note: Metrics computed on validation set using correlations.csv as ground truth. F2-score prioritizes recall as per typical recommendation system evaluation.*

## What Would You Have Done With More Time?

1. **Hybrid Approach**: Combine semantic embeddings with metadata features (content_kind, topic_category) using a learned weighting scheme
2. **Fine-tuning**: Domain-adapt the sentence transformer on K-12 educational corpus
3. **Negative Sampling**: Train a proper cross-encoder re-ranker on positive/negative content-topic pairs
4. **Threshold Optimization**: Per-topic threshold tuning rather than global threshold
5. **Evaluation**: More comprehensive metrics including NDCG, MAP, and per-category performance analysis
6. **Data Augmentation**: Leverage topic hierarchies and content language fields for improved matching


Quick start

1. Create a virtual environment and install dependencies (python <=3.12):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Compatibility note
------------------

Some binary packages (notably PyTorch) are compiled against specific NumPy ABI versions. If you see errors like

```
RuntimeError: Numpy is not available
```

or messages about a module compiled with NumPy 1.x not running with NumPy 2.x, pin NumPy to a 1.x release before installing other packages:

```bash
# from project root
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2,>=1.21"
python -m pip install -r requirements.txt
```

This ensures binary compatibility between PyTorch and NumPy. The `requirements.txt` in this repository already pins NumPy to `<2`.
