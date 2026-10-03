# Spam Message Detection

Classify a text message as **SPAM** or **HAM** (not spam) using Machine Learning (NLP).

## Project structure
```
spam_message_detection/
├── data/sample_spam.csv        # synthetic practice data (3000 rows)
├── generate_sample_data.py     # creates the practice data
├── utils.py                    # text cleaning + data loader
├── train.py                    # trains 4 models, compares, saves best
├── predict.py                  # command-line predictions
├── app.py                      # Streamlit web app
├── models/spam_model.joblib    # saved model (created by train.py)
├── reports/                    # metrics + charts (created by train.py)
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python train.py --data data/sample_spam.csv      # 1. train
python predict.py "You won a free prize, click http://x.com"   # 2. test
streamlit run app.py                              # 3. web app
```

## Use REAL data (do this for your final project)
| Dataset | Size | Where |
|---|---|---|
| **SMS Spam Collection** (best to start) | 5,574 SMS | Kaggle: "SMS Spam Collection Dataset" or UCI ML Repository |
| **Enron Email Spam** | ~30K emails | Kaggle: "Enron spam" |
| **SpamAssassin Public Corpus** | ~6K emails | spamassassin.apache.org |
| **Telegram / YouTube comment spam** | 1K-2K | UCI ML Repository |

Put the file in `data/` and run: `python train.py --data data/spam.csv`
(`utils.load_data` understands the Kaggle `spam.csv` with columns v1/v2 and the UCI tab-separated file.)

## How it works (the 7 steps)
1. **Load data**: label (ham/spam) + text.
2. **Clean text**: lowercase, replace URLs/phone numbers/money with tokens (`urltoken`, `numtoken`, `moneytoken`), remove punctuation.
3. **Split**: 80% train / 20% test, *stratified* so both have the same spam ratio.
4. **Vectorize (TF-IDF)**: turn text into numbers. Words that are frequent in a message but rare overall get high weight. We use 1-word and 2-word phrases ("click here").
5. **Train** 4 models: Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
6. **Evaluate**: 5-fold cross-validation picks the best model; test set gives the final score.
7. **Save + deploy**: save with joblib, use in `predict.py` and `app.py`.

## Metrics (do NOT rely on accuracy alone)
Spam is only ~13% of messages, so "always say ham" gets 87% accuracy and is useless.
- **Precision**: of messages flagged spam, how many really were? (low = real messages get blocked)
- **Recall**: of all real spam, how many did we catch?
- **F1**: balance of the two. Use this to compare models.

Blocking a real message (false positive) is usually worse than letting some spam through, so aim for **high precision**.

## Improvement ideas
- Add features: message length, number of capital letters, number of `!`, has-URL flag.
- Tune hyperparameters with `GridSearchCV`.
- Adjust the decision threshold using `predict_proba`.
- Try a transformer (DistilBERT via Hugging Face) for 98-99%+.
- Handle Hindi/Hinglish messages by training on local data.
