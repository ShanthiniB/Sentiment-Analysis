import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("C:/Users/Shanthini/OneDrive/Desktop/Sentiment Analysis Project/sentiment_analysis.csv")

# Encode categorical columns
label_encoder_time = LabelEncoder()
label_encoder_sentiment = LabelEncoder()
label_encoder_platform = LabelEncoder()
df['Time of Tweet'] = label_encoder_time.fit_transform(df['Time of Tweet'])
df['sentiment'] = label_encoder_sentiment.fit_transform(df['sentiment'])
df['Platform'] = label_encoder_platform.fit_transform(df['Platform'])

# Select input and output columns
X = df["text"]
Y = df["sentiment"]

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Save model and vectorizer with pickle
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder_sentiment, open("label_encoder_sentiment.pkl", "wb"))
