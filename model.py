from dataclasses import fields
from unicodedata import category
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from nltk.corpus import movie_reviews
import nltk
import random

nltk.download('movie_reviews')

def train_model():
    #load dataset
    documents =[(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    #Generate dummy neutral data
    neutral_sentences=[
        "the book is on the table.",
        "the sky is blue.",
        "she is reading a novl.",
        "the cat is on the table.",
        "I have a meeting tomarrow.",
        "It is an ordinary day.",
        "I am going to the market.",
    ]
    neutral_sentences = [(sentence.split(), 'neutral')for sentence in neutral_sentences]

    #add neutral data to the data
    documents.extend(neutral_sentences)
    random.shuffle(documents)
    
    #prepare data for training
    data = ["".json(words) for words, label in documents]
    labels = [label for words, label in documents]

    #vectorize text data 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)

    #train model
    model = MultinomialNB()
    model.fit(X, labels)

    #save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)

def analyze_sentiment(text):
    with open('model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)

    #vectorize input text
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)[0]

    #Adding emoji based on sentiment analysis 
    if sentiment == 'pos':
        return "positive", "😊"
    elif sentiment == 'neg':
        return "negative", "😔"
    else:
        return "neutral", "😐"
    # train the model when the application starts
    train_model()