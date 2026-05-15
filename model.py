import joblib
import re
import string


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# LOAD MODELS
# =========================
def load_models():

    vectorizer = joblib.load("saved_models/vectorizer.pkl")

    nb_model = joblib.load("saved_models/nb_model.pkl")

    lr_model = joblib.load("saved_models/lr_model.pkl")

    svm_model = joblib.load("saved_models/svm_model.pkl")

    metrics = joblib.load("saved_models/metrics.pkl")

    return vectorizer, nb_model, lr_model, svm_model, metrics


# =========================
# URL RISK DETECTION
# =========================
def detect_url_risk(text):

    urls = re.findall(r'https?://\S+|www\.\S+', text)

    if len(urls) == 0:
        return 0

    risk = len(urls) * 20

    return min(risk, 100)


# =========================
# PREDICT FUNCTION
# =========================
def predict(text, vectorizer, nb_model, lr_model, svm_model):

    cleaned = clean_text(text)

    vectorized = vectorizer.transform([cleaned])

    # Naive Bayes
    nb_prob = nb_model.predict_proba(vectorized)[0][1] * 100

    # Logistic Regression
    lr_prob = lr_model.predict_proba(vectorized)[0][1] * 100

    # SVM
    svm_pred = svm_model.predict(vectorized)[0]

    # URL score
    url_score = detect_url_risk(text)

    # Final prediction
    final = False

    if nb_prob > 50 or lr_prob > 50 or svm_pred == 1:
        final = True

    return nb_prob, lr_prob, url_score, final