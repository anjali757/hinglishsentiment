#!/usr/bin/env python
# coding: utf-8

# In[39]:


# code


# In[40]:


import nltk
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[41]:


import nltk
nltk.download('stopwords')


# In[42]:


import nltk
nltk.download('punkt')


# In[43]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[44]:



data = pd.read_csv(r"C:\Users\Anjali Rajora\Downloads\project final\merge reviews.csv")


# In[45]:


data.shape


# In[46]:


data.Sentiment.unique()


# In[47]:


val=data.Sentiment.value_counts()


# In[48]:


val


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


plt.bar(['Positive','Negative','Neutral'],val)


# In[51]:


# 2. Preprocessing
stop_words = set(stopwords.words('english'))


# In[52]:


def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        # Remove special characters and numbers 
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        words = word_tokenize(text)
        # Remove stop words
        words = [w for w in words if not w in stop_words]
        return " ".join(words)
    return ""


# In[53]:


data['processed_text'] = data['Review'].apply(preprocess_text)
data.dropna(subset=['processed_text', 'Sentiment'], inplace=True)


# In[54]:


# 3. Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
X = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
y = data['Sentiment']


# In[55]:


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


# 5. Train a Sentiment Analysis Model (Naive Bayes is a good starting point)
'''model = MultinomialNB()
model.fit(X_train, y_train)'''


# In[57]:


# 6. Evaluate the model
'''y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)'''


# In[58]:


'''print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)'''


# In[59]:


# 7. Function to predict sentiment of new Hinglish text
'''def predict_hinglish_sentiment(text):
    processed_text = preprocess_text(text)
    if not processed_text:
        return "Neutral/Unknown"
    text_vectorized = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vectorized)[0]
    return prediction
'''


# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[61]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[62]:



print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[63]:


def predict_hinglish_sentiment(text):
    processed_text = preprocess_text(text)
    if not processed_text:
        return "Neutral/Unknown"
    text_vectorized = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vectorized)[0]
    return prediction


# In[64]:


import joblib
joblib.dump(model,'model.pkl')
joblib.dump(tfidf_vectorizer,'tfidf.pkl')


# In[65]:


new_data = ["ye product bilkul bekar hai", "acha laga mujhe", "theek thaak hai"]
new_data_tfidf = tfidf_vectorizer.transform(new_data)
predictions = model.predict(new_data_tfidf)
print(predictions) 


# In[ ]:





# In[ ]:





# In[66]:


# Example usage:
new_hinglish_text = "yaar yeh movie achi hai!"
sentiment = predict_hinglish_sentiment(new_hinglish_text)
print(f"Sentiment of '{new_hinglish_text}': {sentiment}")


# In[67]:


new_hinglish_text_2 = "bilkul bekar movie, paisa barbaad."
sentiment_2 = predict_hinglish_sentiment(new_hinglish_text_2)
print(f"Sentiment of '{new_hinglish_text_2}': {sentiment_2}")


# In[68]:


text="ye product mujhe mast lga"
sentiment3 = predict_hinglish_sentiment(text)
print(f"Sentiment of '{text}': {sentiment3}")


# In[69]:


text1="ye product bekar hai"
sentiment4 = predict_hinglish_sentiment(text1)
print(f"Sentiment of '{text1}': {sentiment4}")


# In[70]:


text2="yeh product thik hai but quality achi ho skti thi"
sentiment5 = predict_hinglish_sentiment(text2)
print(f"Sentiment of '{text2}': {sentiment5}")


# In[ ]:




