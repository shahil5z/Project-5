import pandas as pd
import re

# Load the datasets
data = pd.read_csv(r'C:\Users\SHAHIL\Downloads\Project\LLM\feedback.csv')

# Preview the data
print("review")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Check the data types
print("Data types of each column:")
print(data.dtypes)

# Check the distribution of sentiment labels
print("Sentiment Distribution:")
print(data['Sentiment'].value_counts())

import matplotlib.pyplot as plt

# Plot sentiment distribution
data['Sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Preprocess the Data (Clean the 'Review Text' column)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Apply the cleaning function to the 'Review Text' column
data['Cleaned Review Text'] = data['Review Text'].apply(clean_text)

# Preview cleaned data
print("Cleaned Review Text Sample:")
print(data['Cleaned Review Text'].head())

# Convert Sentiment Labels (Map sentiments to numerical labels)

# Map sentiments to numerical values
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['Sentiment Label'] = data['Sentiment'].map(sentiment_mapping)

# Preview the updated dataset
print("Updated Dataset Sample with Sentiment Labels:")
print(data.head())

# Split the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['Cleaned Review Text'], data['Sentiment Label'], test_size=0.2, random_state=42
)

# Vectorize the Text Data using TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=500)

# Fit and transform the training data, and transform the test data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Model (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=sentiment_mapping.keys()))


# Save the Model and Vectorizer
import joblib

# Specify the directory where you want to save the model and vectorizer
save_directory = r'C:\Users\SHAHIL\Downloads\Project\LLM'

# Save the model and vectorizer with the full path
joblib.dump(model, f'{save_directory}\\sentiment_model.pkl')
joblib.dump(vectorizer, f'{save_directory}\\vectorizer.pkl')

print("Model and vectorizer saved in", save_directory)

# Step 10: Test the Model with New Feedback

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test feedback
new_feedback = ["The flight was delayed and staff were rude."]
cleaned_feedback = [clean_text(feedback) for feedback in new_feedback]
vectorized_feedback = vectorizer.transform(cleaned_feedback)

# Predict sentiment
predicted_sentiment = model.predict(vectorized_feedback)
sentiment_label = {v: k for k, v in sentiment_mapping.items()}
print("Predicted Sentiment:", sentiment_label[predicted_sentiment[0]])
