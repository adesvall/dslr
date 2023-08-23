import os
import pickle
import pathlib
import argparse
import numpy as np
from model import Model
from sys import stderr
from data_set import DataSet
from matplotlib import pyplot as plt


def normalize(X):
    return (X - X.mean()) / X.std()


def plot_data(x, y, colors, colormap, features, labels, i):
    plt.subplot(2, 4, i)
    plt.scatter(x[:, 0], x[:, 1], color=colors)
    plt.xlabel(features[0] + ' (normalized)')
    plt.ylabel(features[1] + ' (normalized)')
    plt.grid(True)

    plt.subplot(2, 4, i + 1)
    plt.scatter(x[:, 2], x[:, 3], color=colors)
    plt.xlabel(features[2] + ' (normalized)')
    plt.ylabel(features[3] + ' (normalized)')
    plt.grid(True)

    plt.subplot(2, 4, i + 2)
    plt.scatter(x[:, 4], x[:, 5], color=colors)
    plt.xlabel(features[4] + ' (normalized)')
    plt.ylabel(features[5] + ' (normalized)')
    plt.grid(True)

    legend_handles = []
    legend_labels = []
    for value in np.unique(y):
        val = int(value)
        dummy_handle = plt.scatter([], [], color=colormap[val])
        legend_handles.append(dummy_handle)
        legend_labels.append(labels[val])

    plt.subplot(2, 4, i + 3)
    plt.axis('off')
    plt.legend(legend_handles, legend_labels, title='Hogwarts House', loc='lower left')


def main():
    parser = argparse.ArgumentParser(
        description='Use Logistic Regression to predict Hogwarts House'
    )

    parser.add_argument('dataset', help='path to the csv dataset', type=pathlib.Path)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--log', action='store_true', help='plot loss history')
    parser.add_argument('-i', '--iter', type=int, choices=range(1, 1000000), metavar='N', help='number of iterations for gradient descent (default 5000)', default=5000)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-b', '--batch', type=int, choices=range(1, 1200), help='batch size for mini-batch GD (range[1, 1200])', metavar='N')
    group.add_argument('-s', '--stochastic', action='store_true', help='stochastic gradient descent')

    args = parser.parse_args()
    if args.stochastic:
        args.batch = 1

    try:
        data_set = DataSet(args.dataset)
    except Exception as e:
        print(e, file=stderr)
        return

    features = ['Ancient Runes', 'Herbology', 'Charms', 'Flying', 'Astronomy', 'Potions']
    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    try:
        data_set.df = data_set.df.dropna()
        dfX = normalize(data_set.df[features])
        X = np.array(dfX).astype(float)
        Y = np.array(data_set.df['Hogwarts House']).reshape(-1, 1)

        train_x, test_x, train_y, test_y = DataSet.split(X, Y, 0.5)
        train_x, test_x = train_x.astype(float), test_x.astype(float)

        models = [Model(np.random.random(len(features) + 1).reshape(-1, 1), max_iter=args.iter, batch=args.batch, log=args.log) for _ in range(len(labels))]

        for i, label in enumerate(labels):
            train_y_mapped = (train_y == label).astype(int).reshape(-1, 1)

            if (args.verbose):
                print(f'Training model for label {label}...')
            models[i].fit(train_x, train_y_mapped)
            models[i].predictions = models[i].predict(test_x)
            if (args.verbose):
                print(f'Training done\nLoss: {models[i].loss((test_y == label).astype(int).reshape(-1, 1), models[i].predictions)}\n')

        predictions = np.argmax(np.concatenate([model.predictions for model in models], axis=1), axis=1)
        test_y_mapped = np.searchsorted(labels, test_y)
    except:
        print('Error: couldnt do the training on this dataset\nMake sure it contains the necessary features', file=stderr)
        return

    accuracy = Model.accuracy(test_y_mapped, predictions)
    if (args.verbose):
        print(f'Correct predictions: {np.count_nonzero(predictions == test_y_mapped)} / {len(predictions)}')
        print(f'Accuracy: {accuracy * 100:.5f}%')

    colormap = ['red', 'yellow', 'blue', 'green']
    label_colors = [colormap[code] for code in test_y_mapped]
    prediction_colors = [colormap[code] for code in predictions]

    try:
        os.makedirs('plots', exist_ok=True)
    except:
        print('Error: couldnt create plot directory', file=stderr)
        return

    plt.figure(figsize=[22, 10])

    plot_data(test_x, test_y_mapped, label_colors, colormap, features, labels, 1)
    plot_data(test_x, predictions, prediction_colors, colormap, features, labels, 5)
    plt.savefig('plots/predictions.png')
    if args.verbose:
        plt.show()

    if args.log:
        plt.figure(figsize=[20, 10])
        for i, label in enumerate(labels):
            plt.subplot(2, 2, i + 1)
            plt.plot(range(models[i].max_iter), models[i].loss_hist, color=colormap[i])
            plt.title(f'{label} model')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)

        plt.savefig('plots/loss_history.png')
        if args.verbose:
            plt.show()

    with open('thetas.pickle', 'wb') as handle:
        for model in models:
            pickle.dump(model.thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
