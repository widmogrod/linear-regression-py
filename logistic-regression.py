# from toolz import curry
import csv

def get_data():
    with open("./data/question_quality_classification.csv") as f:
    # with open("./data/question_quality_classification_small.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

from random import shuffle

def split_data(data, test_percentage=0.2):
    total = len(data)
    shuffle(data)
    index = round(total * test_percentage)

    return (data[index:], data[0:index])


data = list(get_data())
train_data, test_data = split_data(data)

def get_column(reader, column):
    for row in reader:
        yield row[column]


from sklearn.feature_extraction.text import CountVectorizer

bag_of_words = CountVectorizer(max_features=10000)
X = bag_of_words.fit_transform(get_column(train_data, 'content'))

# X = X.toarray()
# print(X.shape)
# vocabe = bag_of_words.get_feature_names()
# print(vocabe)


y = list(get_column(train_data, 'moderated_pos'))

import numpy as np
from sklearn import linear_model

logistic = linear_model.LogisticRegression()
logistic.fit(X, y)

print(logistic.coef_)


import matplotlib.pyplot as plt

X_test = bag_of_words.transform(get_column(test_data, 'content'))
y_test = logistic.predict(X_test)
y_expected = list(get_column(test_data, 'moderated_pos'))


ok = [(a == b) for a, b in list(zip(y_expected, y_test))]
ok = [k for k in ok if k == True]
ok = len(ok)/len(test_data) * 100

print('ok:   ', ok)
print('from: ', len(test_data))


plt.savefig('logistic.png')

