from collections import Counter
import pickle
import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding, Flatten, Bidirectional
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Input, Model
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def load_glove(file):
    f = open(file, 'r')
    embed = []
    word = []
    pair = dict()
    for line in f:
        split = line.split()
        word.append(split[0])
        embedding = np.array([float(val) for val in split[1:]])
        embed.append(embedding)
        pair[split[0]] = embedding
    embed = np.array(embed)
    return pair,word

df = pd.read_csv('fake_or_real_news.csv')
b = pd.read_csv('project-dataset-1.csv')
#df = pd.concat([a,b])
df = shuffle(df)
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())
#normalization true casing
for i in df.title.str.split('\n'):
    i[0] = i[0].lower()
for i in df.text.str.split('\n'):
    i[0] = i[0].lower()

df.title = df.title.str.replace(r'\s\s+',' ') #replace multple white space with a single one
df.text = df.text.str.replace(r'\s\s+',' ')
df.title = df.title.str.strip()
df.text = df.text.str.strip()


glove = 'glove.6B.50d.txt'
glove_vec,words = load_glove(glove)

#print(glove_vec['newyork'])
#only considering words that appear for more than 5 times
all_words = ' '.join(df.text.values)
word = all_words.split()
common_words = Counter(word).most_common()
common_word = [p for p,q in common_words]
freq_words = [w[0] for w in common_words if w[1]>5]
#we take all words in having glove vectors and freq words not iin glove
common_words_in_glove = [w for w,is_true in zip(common_word,words) if is_true]
common_words_not_in_glove = [w for w,is_true in zip(common_word,words) if not is_true]
freq_words_not_in_glove = [w for w in common_words_not_in_glove and freq_words]
total_words = common_words_in_glove+freq_words_not_in_glove
word2num = dict(zip(total_words,range(len(total_words))))
word2num['Other'] = len(word2num)
word2num['Pad'] = len(word2num)
num2word = dict(zip(word2num.values(),word2num.keys()))
text = [[word2num[word] if word in word2num else word2num['Other']
             for word in content.split()] for content in df.text.values]
#padding since each x is not of same length
for i,t in enumerate(text):
    if (len(t) < 500):
        text[i] = [word2num['Pad']] * (500 - len(t)) + t #append to the list the pad word's list 800-len(t) times
        #print(len(text[i]))
    elif (len(t) > 500):
        text[i] = t[:500]
        #print(len(text[i]))
    else:
        continue

#print(w)
x = np.array(text,dtype=object)
y = (df.label.values=='REAL').astype('int')
batch_size = 100
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=42)
model = Sequential()
model.add(Embedding(len(word2num),batch_size))
model.add(Bidirectional(LSTM(64),merge_mode='concat'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="AdaGrad",metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size= batch_size, epochs = 10,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)
print(model.predict(x_test),' ',y_test)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

