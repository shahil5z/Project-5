About This Project:

This project is about analyzing customer feedback to figure out if the sentiment (feeling) is Positive, Neutral or Negative.
It uses machine learning to do the analysis and has a web app built with Flask where users can try it out.
The goal is to make it easy for businesses to understand what customers feel about their products or services.

What Tools Were Used:
-Python for coding
-Libraries: Pandas for data cleaning, scikit-learn for machine learning, Flask for the web app, joblib to save the model

What This Project Can Do:

1.Understand Feedback Data
-Clean the feedback text (remove unnecessary characters, make text lowercase etc.)
-Turn text into numbers using a method called TF-IDF.

2.Build a Model
-Train a machine learning model to recognize sentiment (Positive, Neutral, Negative).
-Save the trained model so it can be used later.

3.Web App Features
-Users can type feedback into a form and see the sentiment prediction.
-Show example feedback and their sentiment predictions.

4.User-Friendly Design
-Simple and clean interface for beginners to use.

How to Use This Project
-Install Python and the required libraries by running this command: pip install pandas scikit-learn flask joblib
-Train the Model
-Download the feedback.csv dataset
-Run the feedback.py code to create the model and vectorizer files (sentiment_model.pkl and vectorizer.pkl)
-Run the app.py code
