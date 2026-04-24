import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset (FIXED PATH)
data_path = os.path.join(BASE_DIR, "data", "language_dataset.csv.xlsx")
data = pd.read_excel(data_path)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data["clean_text"] = data["Text"].apply(preprocess)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(data["clean_text"])
y = data["Language"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb = MultinomialNB()
nb.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pred_nb = nb.predict(X_test)
pred_lr = lr.predict(X_test)

print("Naive Bayes Results")
print(classification_report(y_test, pred_nb))

print("Logistic Regression Results")
print(classification_report(y_test, pred_lr))

# Save models (FIXED PATHS)
model_dir = os.path.join(BASE_DIR, "models")

pickle.dump(nb, open(os.path.join(model_dir, "language_model_nb.pkl"), "wb"))
pickle.dump(lr, open(os.path.join(model_dir, "language_model_lr.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(model_dir, "vectorizer.pkl"), "wb"))