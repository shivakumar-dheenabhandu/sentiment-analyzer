# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , LSTM , Embedding
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import pickle

# %%
data = pd.read_csv("Reviews.csv")
data.head()

# %%
data.shape

# %%
data.isna().sum().to_frame(name='# of missing values')

# %%
total_rows=data.shape[0]
data.dropna(how='any',inplace=True)
remaining_rows=data.shape[0]
removed_rows=total_rows-remaining_rows
print("No. of rows removed :",removed_rows)
print(f"\nPercentage of data removed:{np.round((removed_rows/total_rows)*100,2)}%")
print(f"Percentage of data remaining:{np.round((remaining_rows/total_rows)*100,2)}%")

# %%
a =  data.shape[0]
data.drop_duplicates(inplace=True, subset=['Score','Text'])
b = data.shape[0]
print("No. of rows removed :", a-b)
print(f"\nPercentage of data removed: {np.round(((a-b)/total_rows)*100,2)}%")
print(f"Percentage of data remaining: {np.round((b/total_rows)*100,2)}%")

# %%
a=data.shape[0]
idx=data[data["HelpfulnessNumerator"]>data["HelpfulnessDenominator"]].index
data.drop(index=idx, inplace=True)
b=data.shape[0]
print("No. of rows removed :", a-b)
print(f"\nPercentage of data removed:{np.round(((a-b)/total_rows)*100,2)}%")
print(f"Percentage of data remaining:{np.round((b/total_rows)*100,2)}%")

# %%
def create_target(x):
    return "Positive" if x>3 else "Negative" if x<3 else "Neutral"
data.loc[:,'target']=data.Score.apply(create_target)

# %%
data[['Score','target']].sample(5)

# %%
fig, ax = plt.subplots(figsize=(16,6))
vc = data.target.value_counts()
vc.plot.barh(color="blue",fontsize=14,ax=ax)
ax.set_title("Label vs Count", fontsize=15)
plt.show()

# %%
neutral=data.loc[data.target=="Neutral"] 
positive=data.loc[data.target=="Positive"].sample(50000)
negative=data.loc[data.target=="Negative"].sample(50000)
data=pd.concat([positive, negative, neutral])
data.shape

# %%
fig, ax=plt.subplots(figsize=(16, 6))
vc=data.target.value_counts()
vc.plot.barh(color="blue",fontsize=14,ax=ax)
ax.set_title("Label vs Count",fontsize=15)
plt.show()

# %%
total_stopwords=set(stopwords.words('english'))
negative_stop_words=set(word for word in total_stopwords 
                          if "n't" in word or 'no' in word)
final_stopwords=total_stopwords-negative_stop_words
final_stopwords.add("one")
print(final_stopwords)

# %%
stemmer = PorterStemmer()
HTMLTAGS = re.compile('<.*?>')
table = str.maketrans(dict.fromkeys(string.punctuation))
remove_digits = str.maketrans('', '', string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")

# %%
def preprocessor(review):
    review = HTMLTAGS.sub(r'', review)
    review = review.translate(table)
    review = review.translate(remove_digits)
    review = review.lower()
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    review = [word for word in review.split()
              if word not in final_stopwords]
    review = ' '.join([stemmer.stem(word) for word in review])
    return review

# %%
print("Before preprocessing : ")
data.Text.iloc[6]

# %%
data.Text=data.Text.apply(preprocessor) 
print("After preprocessing : ")
data.Text.iloc[6]

# %%
def generate_wcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color='white')
    wordcloud.generate(text)
    plt.figure(figsize=(15,7))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    return plt.show()

# %%
pos = data.loc[data.target=="Positive"].Text
text = " ".join(review for review in pos.astype(str))
generate_wcloud(text)

# %%
pos = data.loc[data.target=="Negative"].Text
text = " ".join(review for review in pos.astype(str))
generate_wcloud(text)

# %%
pos = data.loc[data.target=="Neutral"].Text
text = " ".join(review for review in pos.astype(str))
generate_wcloud(text)

# %%
X = data.Text
y = data.target
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.20, random_state=1, stratify=y)

# %%
X_train.shape, X_test.shape

# %%
bow_vectorizer = CountVectorizer(max_features=10000)
bow_vectorizer.fit(X_train)
bow_X_train = bow_vectorizer.transform(X_train)
bow_X_test = bow_vectorizer.transform(X_test)

# %%
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
tfidf_vectorizer.fit(X_train)
tfidf_X_train = tfidf_vectorizer.transform(X_train)
tfidf_X_test = tfidf_vectorizer.transform(X_test)

# %%
labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)
labels = labelEncoder.classes_.tolist()
print(labels) 

# %%
def train_and_eval(model, trainX, trainY, testX, testY):
    _ = model.fit(trainX, trainY)
    y_preds_train = model.predict(trainX)
    y_preds_test = model.predict(testX)
    print()
    print(model)
    print(f"Train accuracy score : {accuracy_score(y_train, y_preds_train)}")
    print(f"Test accuracy score : {accuracy_score(y_test, y_preds_test)}")
    print('\n',40*'-')

# %%
C = [0.001, 0.01, 0.1, 1, 10]
for c in C: 
    log_model = LogisticRegression(C=c, max_iter=500, random_state=1)
    train_and_eval(model=log_model,
                   trainX=bow_X_train,
                   trainY=y_train,
                   testX=bow_X_test,
                   testY=y_test)

# %%
alphas = [0, 0.2, 0.6, 0.8, 1]
for a  in alphas: 
    nb_model = MultinomialNB(alpha=a)
    train_and_eval(model=nb_model,
                   trainX=bow_X_train,
                   trainY=y_train,
                   testX=bow_X_test,
                   testY=y_test)

# %%
C = [0.001, 0.01, 0.1, 1, 10]
for c in C: 
    log_model = LogisticRegression(C=c, max_iter=500, random_state=1)
    train_and_eval(model=log_model,
                   trainX=tfidf_X_train,
                   trainY=y_train,
                   testX=tfidf_X_test,
                   testY=y_test)

# %%
alphas = [0, 0.2, 0.6, 0.8, 1]
for a  in alphas: 
    nb_model = MultinomialNB(alpha=a)
    train_and_eval(model=nb_model,
                   trainX=tfidf_X_train,
                   trainY=y_train,
                   testX=tfidf_X_test,
                   testY=y_test)

# %%
def plot_cm(y_true, y_pred):
    plt.figure(figsize=(6,6))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(
        cm, annot=True, cmap='Blues', cbar=False, fmt='.2f',
        xticklabels=labels, yticklabels=labels)
    return plt.show()

# %%
bmodel = LogisticRegression(C=1, max_iter=500, random_state=1)
bmodel.fit(tfidf_X_train, y_train)

# %%
y_preds_train = bmodel.predict(tfidf_X_train)
y_preds_test = bmodel.predict(tfidf_X_test)

# %%
print(f"Train accuracy score : {accuracy_score(y_train, y_preds_train)}")
print(f"Test accuracy score : {accuracy_score(y_test, y_preds_test)}")

# %%
plot_cm(y_test, y_preds_test)

# %%
import pickle

# %%
with open("transformer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)    
with open("model.pkl", "wb") as f:
    pickle.dump(bmodel, f)

# %%
def get_sentiment(review):
    x = preprocessor(review)
    x = tfidf_vectorizer.transform([x])
    y = int(bmodel.predict(x.reshape(1,-1)))
    return labels[y]

# %%
review = "This chips packet is very tasty. I highly recommend this!"
print(f"This is a {get_sentiment(review)} review!")

# %%
review = "This product is a waste of money. Don't buy this!!"
print(f"This is a {get_sentiment(review)} review!")


