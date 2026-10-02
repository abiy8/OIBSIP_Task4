# Spam Message Classification

Classify text messages as spam or ham using bag-of-words features and a multinomial Naive Bayes model.

**Stack:** Python · pandas · NumPy · scikit-learn · Seaborn · Jupyter

## Workflow

1. Load the included `spam.csv` with `Message` and `Category` columns.
2. Reserve 25% of messages for testing.
3. Fit `CountVectorizer` on the training messages only.
4. Train `MultinomialNB`, report test accuracy, and predict sample messages.

## Run locally

```bash
git clone https://github.com/abiy8/spam-message-classifier.git
cd spam-message-classifier
python -m venv .venv
# Activate .venv for your operating system.
pip install jupyter pandas numpy scikit-learn seaborn
jupyter notebook
```

Open `OIBSIP_Task4.ipynb` and run cells in order.

## Project context and limitations

This is an Oasis Infobyte task / learning project. The train/test split has no fixed random seed, so results can vary between runs. Accuracy alone does not describe spam detection quality; future evaluation should include precision, recall, and stratified splitting. The example messages demonstrate inference, not a deployed email filter.

## Reproduced portfolio evaluation

A separate baseline evaluation on 2 October 2026 used the existing `CountVectorizer` + `MultinomialNB` approach with exact-message deduplication and a reproducible stratified split. It did not change this notebook's code or saved outputs.

- Input: 5,572 messages; 415 duplicate messages removed; 5,157 unique messages retained.
- Split: 75% training (3,867 messages), 25% holdout (1,290 messages), stratified by category, `random_state=42`.
- Holdout accuracy: **97.9%**; spam precision: **95.9%**; spam recall: **86.9%**; spam F1: **91.1%**.
- Confusion matrix (actual rows / predicted columns: ham, spam): `[[1124, 6], [21, 139]]`.

These are results from one deduplicated holdout, not cross-validation or production performance.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('spam.csv').drop_duplicates(subset=['Message'])
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Category'], test_size=0.25,
    random_state=42, stratify=df['Category']
)
model = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('accuracy', accuracy_score(y_test, pred))
print('spam precision/recall/F1', precision_recall_fscore_support(
    y_test, pred, average='binary', pos_label='spam', zero_division=0
)[:3])
```
