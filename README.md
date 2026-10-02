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
git clone https://github.com/abiy8/OIBSIP_Task4.git
cd OIBSIP_Task4
python -m venv .venv
# Activate .venv for your operating system.
pip install jupyter pandas numpy scikit-learn seaborn
jupyter notebook
```

Open `OIBSIP_Task4.ipynb` and run cells in order.

## Project context and limitations

This is an Oasis Infobyte task / learning project. The train/test split has no fixed random seed, so results can vary between runs. Accuracy alone does not describe spam detection quality; future evaluation should include precision, recall, and stratified splitting. The example messages demonstrate inference, not a deployed email filter.
