import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import tensorflow as tf
import numpy as np

# Download necessary NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function (same as training)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Load saved TF-IDF vectorizer, label encoder, and Logistic Regression model
tfidf = joblib.load(r'C:\Users\LEGION\tfidf_vectorizer.pkl')
le = joblib.load(r'C:\Users\LEGION\label_encoder.pkl')
lr_model = joblib.load(r'C:\Users\LEGION\logistic_regression_model.pkl')

# Load saved Neural Network model
nn_model = tf.keras.models.load_model(r"C:\Users\LEGION\neural_network_model.h5")

def predict_logistic_regression(text):
    clean_text = preprocess_text(text)
    vectorized = tfidf.transform([clean_text])
    pred_index = lr_model.predict(vectorized)[0]
    pred_label = le.inverse_transform([pred_index])[0]
    return pred_label

def predict_neural_network(text):
    clean_text = preprocess_text(text)
    vectorized = tfidf.transform([clean_text]).toarray()
    pred_probs = nn_model.predict(vectorized)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    pred_label = le.inverse_transform([pred_index])[0]
    return pred_label

if __name__ == "__main__":
    print("Enter a news article (title + description):")
    user_input = input()

    print("\nPredicting category with Logistic Regression...")
    lr_pred = predict_logistic_regression(user_input)
    print(f"Logistic Regression Prediction: {lr_pred}")

    print("\nPredicting category with Neural Network...")
    nn_pred = predict_neural_network(user_input)
    print(f"Neural Network Prediction: {nn_pred}")
