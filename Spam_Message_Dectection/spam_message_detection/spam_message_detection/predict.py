"""Classify messages with the trained model.
Usage:  python predict.py "You won a free prize, click http://x.com"
        python predict.py            (interactive mode)
"""
import sys, joblib
model = joblib.load("models/spam_model.joblib")

def classify(msg):
    label = model.predict([msg])[0]
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        score = f"{model.predict_proba([msg])[0][1]:.1%} spam probability"
    else:
        score = f"decision score {model.decision_function([msg])[0]:+.2f} (>0 = spam)"
    return ("SPAM" if label else "HAM (not spam)"), score

if len(sys.argv) > 1:
    for m in sys.argv[1:]:
        print(m, "->", *classify(m))
else:
    print("Type a message (empty line to quit)")
    while (m := input("> ").strip()):
        print("  ->", *classify(m))
