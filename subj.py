import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import nltk
import numpy as np
#nltk.download('stopwords')
'exec(%matplotlib inline)'
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
amz_reviews = pd.read_csv("7817_1.csv")

columns = []
df = pd.DataFrame(amz_reviews.drop(columns,axis=1,inplace=False))

plt.figure(figsize=(20,17))
df['reviews.rating'].value_counts()[:].plot(kind='bar')
#plt.show()

## Change the reviews type to string
df['reviews.text'] = df['reviews.text'].astype(str)

## Before lowercasing
print("Before Lowercasing:")
print(df['reviews.text'][1:])

## Lowercase all reviews
print("After Lowercasing:")
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(df['reviews.text'][1:])

stop = stopwords.words('english')
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print("After removing stopwords:")
print(df['reviews.text'][1:])

st = PorterStemmer()
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
print("After stemming:")
print(df['reviews.text'][1:])

def senti(x):
    return TextBlob(x).sentiment  

df['senti_score'] = df['reviews.text'].apply(senti)
#df.senti_score.to_csv('senti.csv',index=False)

df['polarity'] = df.apply(lambda x: TextBlob(x['reviews.text']).sentiment.polarity, axis=1)
df['subjectivity'] = df.apply(lambda x: TextBlob(x['reviews.text']).sentiment.subjectivity, axis=1)

print("printing polarity to file polar.csv")
df.polarity.to_csv('polar.csv',index=False)

print("printing subjectivity to subj.csv")
df.subjectivity.to_csv('subj.csv',index=False)

print("printing reviews after NLP")
#df.reviewsto_csv(['reviews.text'][2],index=False)
df[['reviews.rating' , 'reviews.text']].to_csv('reviews.csv',index=False)


