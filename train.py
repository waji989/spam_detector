import os
import pandas as pd
import joblib
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create models directory
os.makedirs("saved_models", exist_ok=True)

print("="*50)
print("📧 SPAM DETECTION MODEL TRAINER")
print("="*50)

# =========================
# 📂 LOAD DATASET
# =========================
print("\n📂 Loading dataset...")

# Load your specific dataset
df = pd.read_csv("combined_data.csv")  # Your file name

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")
print(f"\n📊 First few rows:")
print(df.head())

# =========================
# 🔍 VERIFY COLUMNS
# =========================
if 'label' not in df.columns or 'text' not in df.columns:
    raise Exception("❌ Dataset must have 'label' and 'text' columns")

# Check label values
print(f"\n📊 Label values: {df['label'].unique()}")
print(f"   - 1 = Spam")
print(f"   - 0 = Ham (Not Spam)")

# =========================
# 🧹 CLEAN TEXT FUNCTION
# =========================
def clean_text(text):
    """Clean and preprocess email text"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

# =========================
# 🧹 PREPARE DATA
# =========================
print("\n🔍 Preparing data...")

# Select columns
df = df[['label', 'text']].copy()
df.columns = ['label', 'message']

# Clean messages
df['message'] = df['message'].astype(str).apply(clean_text)

# Remove empty messages
df = df[df['message'].str.len() > 0]

# Ensure labels are integers
df['label'] = df['label'].astype(int)

print(f"\n✅ Data prepared successfully!")
print(f"📊 Final shape: {df.shape}")
print(f"📊 Class distribution:")
print(f"   Ham (0 - Not Spam): {(df['label'] == 0).sum():,} messages")
print(f"   Spam (1): {(df['label'] == 1).sum():,} messages")
print(f"\n📝 Sample message (Ham):")
print(f"   {df[df['label']==0]['message'].iloc[0][:150]}...")
print(f"\n📝 Sample message (Spam):")
print(f"   {df[df['label']==1]['message'].iloc[0][:150]}...")

# =========================
# 🔄 TRAIN TEST SPLIT
# =========================
print("\n🔄 Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"📊 Training samples: {len(X_train):,}")
print(f"📊 Testing samples: {len(X_test):,}")

# =========================
# 🧠 TF-IDF VECTORIZATION
# =========================
print("\n🧠 Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,  # Increased for better accuracy
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Feature matrix shape: {X_train_vec.shape}")
print(f"   Features: {X_train_vec.shape[1]:,}")

# =========================
# 🤖 TRAIN MODELS
# =========================
print("\n🤖 Training models...")

# 1. Naive Bayes
print("\n📊 Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"✅ Naive Bayes Accuracy: {nb_acc:.4f} ({nb_acc*100:.2f}%)")

# 2. Logistic Regression
print("\n📊 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"✅ Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# 3. SVM
print("\n📊 Training SVM...")
svm_model = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced', dual=True)
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"✅ SVM Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

# =========================
# 📊 DETAILED RESULTS
# =========================
print("\n" + "="*50)
print("📊 DETAILED CLASSIFICATION RESULTS")
print("="*50)

print("\n📈 Naive Bayes Classification Report:")
print(classification_report(y_test, nb_pred, target_names=['Ham (0)', 'Spam (1)']))

print("\n📈 Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['Ham (0)', 'Spam (1)']))

print("\n📈 SVM Classification Report:")
print(classification_report(y_test, svm_pred, target_names=['Ham (0)', 'Spam (1)']))

# =========================
# 💾 SAVE MODELS
# =========================
print("\n💾 Saving models...")

joblib.dump(vectorizer, "saved_models/vectorizer.pkl")
joblib.dump(nb_model, "saved_models/nb_model.pkl")
joblib.dump(lr_model, "saved_models/lr_model.pkl")
joblib.dump(svm_model, "saved_models/svm_model.pkl")

# Save metrics
metrics = {
    'Naive Bayes': nb_acc,
    'Logistic Regression': lr_acc,
    'SVM': svm_acc,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': X_train_vec.shape[1]
}
joblib.dump(metrics, "saved_models/metrics.pkl")

# Save dataset info
dataset_info = {
    'total_samples': len(df),
    'ham_count': int((df['label'] == 0).sum()),
    'spam_count': int((df['label'] == 1).sum()),
    'features': X_train_vec.shape[1]
}
joblib.dump(dataset_info, "saved_models/dataset_info.pkl")

print("\n" + "="*50)
print("✅ TRAINING COMPLETE!")
print("="*50)
print(f"\n📁 Files saved in 'saved_models/' folder:")
print(f"   ✓ vectorizer.pkl (TF-IDF vectorizer)")
print(f"   ✓ nb_model.pkl (Naive Bayes)")
print(f"   ✓ lr_model.pkl (Logistic Regression)")
print(f"   ✓ svm_model.pkl (SVM)")
print(f"   ✓ metrics.pkl (Model performance)")
print(f"   ✓ dataset_info.pkl (Dataset statistics)")

print(f"\n🎯 Best Model: {max(metrics, key=metrics.get)} with {max(nb_acc, lr_acc, svm_acc)*100:.2f}% accuracy")

print("\n🚀 You can now run:")
print("   📱 Web App: streamlit run app_web.py")
print("   💻 Desktop App: python app_gui_modern.py")