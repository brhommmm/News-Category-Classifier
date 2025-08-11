import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# For NN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------
# Data Loading
# ----------------------
def load_data(filepath):
    try:
        return pd.read_csv(filepath, header=None, names=['class_index', 'title', 'description'])
    except:
        try:
            return pd.read_csv(filepath, sep='\t', header=None, 
                             names=['class_index', 'title', 'description'])
        except:
            return pd.read_fwf(filepath, header=None, 
                             names=['class_index', 'title', 'description'])

train_path = r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task2\Dataset\train.csv"
test_path = r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task2\Dataset\test.csv"

train_df = load_data(train_path)
test_df = load_data(test_path)

# ----------------------
# Data Cleaning
# ----------------------
def clean_data(df):
    df = df.copy()
    df['class_index'] = pd.to_numeric(df['class_index'], errors='coerce')
    df = df.dropna(subset=['class_index'])
    df['class_index'] = df['class_index'].astype(int)
    df = df[df['class_index'].between(1, 4)].copy()
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    return df

train_df = clean_data(train_df)
test_df = clean_data(test_df)

# Map categories
category_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
train_df['category_name'] = train_df['class_index'].map(category_map)
test_df['category_name'] = test_df['class_index'].map(category_map)

# Combine text
train_df['text'] = train_df['title'] + " " + train_df['description']
test_df['text'] = test_df['title'] + " " + test_df['description']

# ----------------------
# Text Preprocessing
# ----------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

# ----------------------
# Visualizations
# ----------------------
# Class distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='category_name', palette='viridis')
plt.title('Training Data Class Distribution')
plt.xlabel('News Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# Word clouds
def plot_wordcloud_for_category(df, category_name):
    text = " ".join(df[df['category_name'] == category_name]['clean_text'])
    if not text.strip():
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud: {category_name}")
    plt.tight_layout()
    plt.savefig(f'wordcloud_{category_name.replace("/", "_")}.png')
    plt.close()

for category in train_df['category_name'].unique():
    plot_wordcloud_for_category(train_df, category)

# ----------------------
# Logistic Regression Model
# ----------------------
X_train = train_df['clean_text']
X_test = test_df['clean_text']

le = LabelEncoder()
y_train = le.fit_transform(train_df['category_name'])
y_test = le.transform(test_df['category_name'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = lr.predict(X_test_tfidf)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('logreg_confusion_matrix.png')
plt.close()

# ----------------------
# Neural Network (Bonus)
# ----------------------
print("\nTraining Neural Network...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],), kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train_tfidf.toarray(), y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test_tfidf.toarray(), y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('nn_training_history.png')
plt.show()

# Evaluate NN
nn_loss, nn_acc = model.evaluate(X_test_tfidf.toarray(), y_test, verbose=0)
print(f"\nNeural Network Test Accuracy: {nn_acc:.4f}")

import joblib

# Save TF-IDF Vectorizer and Label Encoder
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Save Logistic Regression model
joblib.dump(lr, 'logistic_regression_model.pkl')

# Save Neural Network model
model.save('neural_network_model.h5')
