#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[11]:


def scrape_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    articles = []
    for article in soup.find_all('article'):  
        title_tag = article.find('h2')
        content_tag = article.find('div', class_='content')
        category_tag = article.find('span', class_='category')
        
        # Check if the tags exist before extracting text
        if title_tag and content_tag and category_tag:
            title = title_tag.text.strip()
            content = content_tag.text.strip()
            category = category_tag.text.strip()
            
            articles.append({'title': title, 'content': content, 'category': category})
    
    return articles


# In[12]:


def classify_news(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['content'])  # Assuming 'content' is the text data
    
    # Assuming 'category' is the label you want to predict
    y = data['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using a simple Naive Bayes classifier as an example
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report


# In[16]:


url = 'https://www.nytimes.com/international/section/sports'
articles = scrape_news(url)

# Store scraped data in a DataFrame
df = pd.DataFrame(articles)

# Classify news articles


# In[17]:


accuracy, report = classify_news(df)

# Save the DataFrame to a CSV file
df.to_csv('scraped_data.csv', index=False)

# Save the classification report to a text file
with open('classification_report.txt', 'w') as file:
    file.write(report)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# In[ ]:





# In[ ]:





# In[ ]:




