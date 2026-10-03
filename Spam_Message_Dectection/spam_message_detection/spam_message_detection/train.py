"""Train and compare spam classifiers, then save the best one.
Usage:  python train.py --data data/sample_spam.csv
        python train.py --data data/spam.csv          (real Kaggle file)
"""
import argparse, json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score, accuracy_score,
                             RocCurveDisplay)
from utils import clean_text, load_data

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample_spam.csv")
args = ap.parse_args()

# ---------- 1. LOAD ----------
df = load_data(args.data)
print(f"Loaded {len(df)} messages | spam share = {df.label.mean():.1%}")

# ---------- 2. SPLIT (stratified keeps spam ratio equal in train/test) ----------
X_train, X_test, y_train, y_test = train_test_split(
    df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)

# ---------- 3. MODELS ----------
def make_pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", clf)])

models = {
    "Naive Bayes":        make_pipe(MultinomialNB(alpha=0.1)),
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Linear SVM":         make_pipe(LinearSVC(class_weight="balanced")),
    "Random Forest":      make_pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                           random_state=42, n_jobs=-1)),
}

# ---------- 4. CROSS-VALIDATE + TEST ----------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
results = {}
print(f"\n{'Model':22s} {'CV-F1':>7s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
for name, pipe in models.items():
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    results[name] = dict(cv_f1=cv_f1, accuracy=accuracy_score(y_test, p),
                         precision=precision_score(y_test, p), recall=recall_score(y_test, p),
                         f1=f1_score(y_test, p))
    r = results[name]
    print(f"{name:22s} {r['cv_f1']:7.3f} {r['accuracy']:7.3f} {r['precision']:7.3f} "
          f"{r['recall']:7.3f} {r['f1']:7.3f}")

# ---------- 5. PICK BEST (by cross-validated F1, NOT by test score) ----------
best_name = max(results, key=lambda k: results[k]["cv_f1"])
best = models[best_name]
print(f"\nBest model: {best_name}")
pred = best.predict(X_test)
print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))

# ---------- 6. SAVE ----------
joblib.dump(best, "models/spam_model.joblib")
json.dump(results, open("reports/metrics.json", "w"), indent=2)

# ---------- 7. PLOTS ----------
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["ham", "spam"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}"); plt.savefig("reports/confusion_matrix.png", dpi=120); plt.close()

if hasattr(best, "decision_function") or hasattr(best, "predict_proba"):
    RocCurveDisplay.from_estimator(best, X_test, y_test); plt.title("ROC curve")
    plt.savefig("reports/roc_curve.png", dpi=120); plt.close()

# Top words that push a message toward spam (works for linear models / NB)
tfidf, clf = best.named_steps["tfidf"], best.named_steps["clf"]
words = np.array(tfidf.get_feature_names_out())
if hasattr(clf, "coef_"): w = clf.coef_[0]
elif hasattr(clf, "feature_log_prob_"): w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
else: w = clf.feature_importances_
top = np.argsort(w)[-15:]
plt.figure(figsize=(7, 5)); plt.barh(words[top], w[top], color="crimson")
plt.title("Top words indicating SPAM"); plt.tight_layout()
plt.savefig("reports/top_spam_words.png", dpi=120); plt.close()
print("Top spam words:", list(words[top][::-1]))

# ---------- 8. SHOW MISTAKES (very useful for improving the model) ----------
wrong = X_test[pred != y_test]
print(f"\n{len(wrong)} misclassified test messages (first 8):")
for t, true in list(zip(wrong, y_test[wrong.index]))[:8]:
    print(f"  [true={'spam' if true else 'ham'}] {t[:90]}")
