from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import os

app = Flask(__name__)

# Paths for the model and vectorizer files
model_path = r'C:\Users\SHAHIL\Downloads\Project\LLM\sentiment_model.pkl'
vectorizer_path = r'C:\Users\SHAHIL\Downloads\Project\LLM\vectorizer.pkl'

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to clean the text input
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Sample feedbacks and their predicted sentiments (for display purposes)
sample_feedbacks = [
    ("The service was excellent and very quick!", "Positive"),
    ("I am not satisfied with the quality of the product.", "Negative"),
    ("The experience was okay, not great but not bad either.", "Neutral"),
    ("Amazing customer support, I will definitely return!", "Positive"),
    ("Very disappointing, I expected much more.", "Negative")
]

# Home route with feedback form
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    feedback = None

    if request.method == 'POST':
        # Get feedback from form
        feedback = request.form.get('feedback')
        if feedback:
            # Clean and vectorize the feedback
            cleaned_feedback = clean_text(feedback)
            vectorized_feedback = vectorizer.transform([cleaned_feedback])

            # Predict sentiment
            predicted_sentiment = model.predict(vectorized_feedback)
            sentiment_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            result = sentiment_label[predicted_sentiment[0]]
    
    # Render the form and display result
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Sentiment Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                color: #333;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                text-align: center;
                width: 400px;
            }
            h1 {
                color: #4CAF50;
            }
            textarea {
                width: 100%;
                height: 100px;
                margin: 10px 0;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                font-size: 18px;
                color: #555;
            }
            .samples {
                margin-top: 30px;
                font-size: 16px;
                color: #333;
            }
            .sample-item {
                margin: 10px 0;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feedback Sentiment Analysis</h1>
            <form method="POST">
                <textarea name="feedback" placeholder="Enter your feedback here..." required>{{ feedback if feedback else '' }}</textarea><br>
                <button type="submit">Analyze Sentiment</button>
            </form>
            {% if result %}
                <div class="result">
                    <strong>Predicted Sentiment:</strong> {{ result }}
                </div>
            {% endif %}
            
            <div class="samples">
                <h2>Sample Predictions</h2>
                {% for feedback, sentiment in sample_feedbacks %}
                    <div class="sample-item">
                        <strong>Feedback:</strong> {{ feedback }}<br>
                        <strong>Predicted Sentiment:</strong> {{ sentiment }}
                    </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    ''', result=result, feedback=feedback, sample_feedbacks=sample_feedbacks)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
