# dataset loaded at
# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=5

# Import the necessary modules
# slightly modified to make it work in my PC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df_path = r'C:/Users/Heng2020/OneDrive/Python NLP/NLP 05_UsefulSenLabel/datacamp code/fake_or_real_news.csv'

df = pd.read_csv(df_path)
# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                        df["text"],
                                        y,  
                                        test_size=0.33,
                                        random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train.values)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=6
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english",max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])
# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=7

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

#print(count_df)
# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A,columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=11
# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# when use TfidfVectorizer directly in MultinomialNB, accuracy is 0.89, but when I convert it to pd.df
# it went down to 0.79

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

# using pd.Df to train rather than  tfidf_vectorizer object
nb_classifier02 = MultinomialNB()
nb_classifier02.fit(count_train,y_train)

X_train_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=X_train.index)
y_train_df = y_train.copy()

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=11
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=14
# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()

# https://campus.datacamp.com/courses/introduction-to-natural-language-processing-in-python/building-a-fake-news-classifier?ex=15
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

test01 = nb_classifier.feature_log_prob_

# Inspecting feature log probabilities
feature_log_probs = nb_classifier.feature_log_prob_

# To get the actual feature names
feature_names = nb_classifier.get_feature_names_out()

# Mapping feature names to their log probabilities
feature_probs = np.exp(feature_log_probs) # converting log probabilities to probabilities
feature_probs_dict = {class_id: dict(zip(feature_names, probs)) for class_id, probs in enumerate(feature_probs)}


# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])


