import pandas as pd
import os

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# create json file
import json

# load data
from sklearn.datasets.base import Bunch

# categorical feature:label
import MultiColumnLabelEncoder
from sklearn.preprocessing import LabelEncoder

# build model
from sklearn.pipeline import Pipeline
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# save the model
import pickle

# *************** Preparing Data ***************
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]
data = pd.read_csv('data/adult.data', names=names)
data.head()


# *************** Data visualization ***************
def visualize_data():
    # print(data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.describe())
    sns.countplot(y='native-country', hue='income', data=data, )
    plt.show()


# *************** Create meta ***************
def create_meta():
    meta = {
        'target_names': list(data.income.unique()),
        'feature_names': list(data.columns),
        'categorical_features': {
            column: list(data[column].unique())
            for column in data.columns
            if data[column].dtype == 'object'
        },
    }
    with open('data/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


# *************** Load data ***************
def load_data(root='data'):
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f)

    names = meta['feature_names']
    # remove some col not needed
    # I thought following is going to improve accuracy but actually it doesn't at all
    # meta['feature_names'].remove('capital-gain')
    # meta['feature_names'].remove('capital-loss')
    # This works and increased overall accuracy
    meta['feature_names'].remove('fnlwgt')
    newnames = meta['feature_names']
    # Load the training and test set
    train = pd.read_csv(os.path.join(root, 'adult.data'), header=0, usecols=newnames, sep=', ', engine='python')
    test = pd.read_csv(os.path.join(root, 'adult.test'), header=0, usecols=newnames, sep=', ', engine='python')

    # Remove the target from the categorical features
    meta['categorical_features'].pop('income')
    return Bunch(
        data=train[newnames[:-1]],
        target=train[newnames[-1]],
        data_test=test[newnames[:-1]],
        target_test=test[newnames[-1]],
        target_names=meta['target_names'],
        feature_names=meta['feature_names'],
        categorical_features=meta['categorical_features'],
    )


# *************** Building model ***************
def build_model(yencode):
    # construct the pipeline
    census = Pipeline([
        ('encode', MultiColumnLabelEncoder.MultiColumnLabelEncoder(dataset.categorical_features)),
        ('classify', RandomForestClassifier(n_estimators=100))
        # ('classify', tree.DecisionTreeClassifier())
        # ('classify', KNeighborsClassifier(n_neighbors=1))
        # ('classify', KNeighborsClassifier(n_neighbors=3))
        # ('classify', KNeighborsClassifier(n_neighbors=5))
        # ('classifier', GaussianNB())
        # ('classifier', LogisticRegression(solver='liblinear'))
    ])
    census.fit(dataset.data, yencode.transform(dataset.target))
    return census


# *************** Save model ***************
def dump_model(model, path='data/model'):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


# *************** Load model ***************
def load_model(path='data/model'):
    with open(path, 'rb') as file:
        return pickle.load(file)


# *************** Predicting ***************
def predict(model):
    data = {}
    with open(os.path.join('data/meta.json'), 'r') as f:
        meta = json.load(f)
    path = 'predict.txt'
    # If has the predict file, then read the file and give prediction
    # otherwise, read from user's input
    if os.path.exists(path):
        file = open(path, 'r')
        line = file.readline()
        attributes = line.split(', ')
        counter = 0
        for column in meta['feature_names'][:-1]:
            data[column] = " " + attributes[counter]
            counter += 1
        file.close()
    else:
        for column in meta['feature_names'][:-1]:
            # Get the valid responses
            valid = meta['categorical_features'].get(column)
            # Prompt the user for an answer until good
            while True:
                val = " " + input("enter {} >".format(column))
                if valid and val not in valid:
                    print("Not valid, choose one of {}".format(valid))
                else:
                    data[column] = val
                    break

    # Create prediction and label
    yhat = model.predict(pd.DataFrame([data]))
    return yencode.inverse_transform(yhat)


if __name__ == "__main__":
    visualize_data()

    print('Loading data ......')
    dataset = load_data()

    print('Building model ......')
    yencode = LabelEncoder().fit(dataset.target)
    census = build_model(yencode)

    print('Testing model ......')
    y_true = yencode.transform([y.rstrip(".") for y in dataset.target_test])
    y_pred = census.predict(dataset.data_test)
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=dataset.target_names))

    print('Saving model ......')
    dump_model(census)

    print()
    print('Loading model ......')
    model = load_model()

    # print('\n*************************')
    # print('Predicting for query instance......')
    # print('Prediction is:', predict(model))
