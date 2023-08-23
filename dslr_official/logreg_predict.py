import csv
import pickle
import pathlib
import argparse
import numpy as np
from sys import stderr
from model import Model
from data_set import DataSet


def normalize(X):
    return (X - X.mean()) / X.std()


def main():
    parser = argparse.ArgumentParser(
        description='Use Logistic Regression to predict Hogwarts House'
    )

    parser.add_argument('dataset', help='path to the csv dataset', type=pathlib.Path)
    parser.add_argument('thetas', help='path to the pickle file containing thetas', type=pathlib.Path)

    args = parser.parse_args()

    try:
        data_set = DataSet(args.dataset)
    except Exception as e:
        print(e, file=stderr)
        return

    features = ['Ancient Runes', 'Herbology', 'Charms', 'Flying', 'Astronomy', 'Potions']
    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    parsed_thetas = []

    with open(args.thetas, 'rb') as handle:
        models = [Model(pickle.load(handle)) for _ in range(len(labels))]

    dfX = normalize(data_set.df[features])
    dfX = dfX.fillna(0)
    X = np.array(dfX).astype(float)

    print('Classifying data...')
    for i, label in enumerate(labels):
        models[i].predictions = models[i].predict(X)
    print('Classifying done')

    predictions = np.argmax(np.concatenate([model.predictions for model in models], axis=1), axis=1)

    with open('houses.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Index', 'Hogwarts House'])
        for i, prediction in enumerate(predictions):
            writer.writerow([i, labels[prediction]])


if __name__ == "__main__":
    main()
