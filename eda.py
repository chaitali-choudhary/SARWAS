import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from wordcloud import WordCloud

def visualise_word_map():
    words=" "
    for msg in dataset["reviews.text"]:
        msg = str(msg).lower()
        words = words+msg+" "
    wordcloud = WordCloud(width=3000, height=2500, background_color='white').generate(words)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 14
    fig_size[1] = 7
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
dataset=pd.read_csv("7817_1.csv")
dataset.describe()
summarised_results = dataset["reviews.rating"].value_counts()

plt.bar(summarised_results.keys(), summarised_results.values)
#,color='blue',edgecolor='yellow')

rt=[0,1,2,3,4,5]
pos=np.arange(len(rt))

plt.xticks(pos,rt )
plt.xlabel('Ratings', fontsize=16)
plt.ylabel('Number of Users', fontsize=16)
#plt.title('Ratings Vs No. of Users',fontsize=20)

plt.show()
visualise_word_map()

