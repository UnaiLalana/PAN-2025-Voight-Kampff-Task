# Generative AI Authorship Verification — PAN 2025 (Voight-Kampff Task 1)

Binary text classification to distinguish **human-written** (`src`) from **AI-generated** (`susp`) text, based on the [PAN 2025 Voight-Kampff shared task](https://pan.webis.de) (Bevendorff et al., 2025).

---

## Repository Structure

```
├── notebooks/
│   └── assignment1.ipynb
├── data/
│   └── reduced_dataset.csv
├── src/
|   └── bevendorff2025.pdf
├── report.tex
├── requirements.txt
└── README.md
```
---

## Dataset

| Split | `susp` (AI) | `src` (Human) | Total |
|-------|------------|--------------|-------|
| Train (80%) | 1,544 | 1,513 | 3,057 |
| Test  (20%) | 385   | 378   | 763   |

**Source:** PAN 2025 official training corpus (Zenodo). Stratified split with `random_state=42`.

---

## Methodology

- **Preprocessing:** lowercasing · punctuation removal · stopword filtering · lemmatisation (NLTK)
- **Sparse features:** TF-IDF (unigrams and bigrams)
- **Dense features:** Word2Vec (dim=100, window=5) → averaged document vectors
- **Classifier:** Logistic Regression with GridSearchCV (`C ∈ {0.01, 0.1, 1, 10}`)
- **Evaluation:** 5-Fold Stratified Cross-Validation · composite score (AUC + Brier + F1 + F0.5u + c@1) / 5

---

## Results

| Representation | CV Score | Test Score |
|----------------|----------|------------|
| TF-IDF Unigrams | 0.845 | — |
| TF-IDF Bigrams (1,2) | 0.862 | 0.856 |
| **Word2Vec (avg)** | **0.879** | **0.910** |

Dense embeddings outperform sparse TF-IDF by **+5.4 pp** on the test set.

---

## Quickstart

Unzip the dataset and place it in the `data` directory.

```bash
unzip reduced_dataset.zip -d data/
pip install -r requirements.txt
jupyter notebook notebooks/assignment1.ipynb
```
---

## Reference

> Bevendorff et al., *Overview of PAN 2025: Voight-Kampff Generative AI Authorship Verification*, CLEF 2025.
