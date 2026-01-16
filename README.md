# Hinglish Sentiment Analysis using NLTK

##  Project Overview

This project focuses on **Sentiment Analysis of Hinglish text** (a mix of Hindi + English).
The model predicts whether a given Hinglish review is **Positive**, **Negative**, or **Neutral**.
It is useful for analyzing:
* Product reviews
* Movie reviews
* User feedback on social media


##  Objective

To build a machine learning model that can:
* Understand Hinglish text
* Clean and preprocess text data
* Convert text into numerical features
* Predict sentiment accurately


## Technologies & Libraries Used

* **Python**
* **Pandas** – data handling
* **NLTK** – text preprocessing (tokenization, stopwords)
* **Scikit-learn** – ML models & evaluation
* **TF-IDF Vectorizer** – feature extraction
* **Logistic Regression** – sentiment classification
* **Matplotlib** – data visualization
* **Joblib** – model saving


##  Project Workflow

1. Load Hinglish review dataset (CSV file)
2. Perform text preprocessing:

   * Convert text to lowercase
   * Remove special characters and numbers
   * Tokenize text
   * Remove stopwords
3. Convert text into numerical form using **TF-IDF**
4. Split data into training and testing sets
5. Train the **Logistic Regression** model
6. Evaluate model using:
   * Accuracy
   * Classification Report
7. Predict sentiment for new Hinglish sentences
8. Save trained model and vectorizer


## Model Performance

* **Algorithm Used:** Logistic Regression
* **Evaluation Metrics:**
  * Accuracy Score
  * Precision, Recall, F1-Score

(The performance depends on dataset size and quality)


##  Example Predictions

```text
Input: "yaar yeh movie achi hai!"
Output: Positive

Input: "bilkul bekar movie, paisa barbaad"
Output: Negative

Input: "yeh product thik hai but quality achi ho skti thi"
Output: Neutral
```


## Saved Files

* `model.pkl` – trained sentiment analysis model
* `tfidf.pkl` – TF-IDF vectorizer
These files can be reused for deployment or future predictions.




##  Future Improvements

* Use Hinglish-specific stopwords
* Try deep learning models (LSTM, BERT)
* Improve neutral sentiment detection




