import os
import pandas as pd
from sys import stderr
from data_set import DataSet
from matplotlib import pyplot as plt


def main():
    try:
        data_set = DataSet('datasets/dataset_train.csv')
    except Exception as e:
        print(e, file=stderr)
        return

    if len(data_set.num_features) == 0:
        print('Error: no numerical features found', file=stderr)

    try:
        os.makedirs('plots', exist_ok=True)
    except:
        print('Error: couldnt create plot directory', file=stderr)
        return

    data_set.df = data_set.df.dropna()

    data = data_set.df[data_set.num_features]
    label_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    colors = [label_colors[house] for house in data_set.df['Hogwarts House']]

    scatter_matrix = pd.plotting.scatter_matrix(data, diagonal='hist', figsize=(24, 12), color=colors)

    for ax in scatter_matrix.ravel():
        ax.xaxis.label.set_rotation(0)
        ax.yaxis.label.set_rotation(0)
        ax.xaxis.labelpad = 25
        ax.yaxis.labelpad = 50

        x_label = ax.xaxis.get_label_text()
        y_label = ax.yaxis.get_label_text()
        if len(x_label) > 10:
            words = x_label.split()
            middle = len(words) // 2
            x_label = '\n'.join([' '.join(words[:middle]), ' '.join(words[middle:])])
        if len(y_label) > 10:
            words = y_label.split()
            middle = len(words) // 2
            y_label = '\n'.join([' '.join(words[:middle]), ' '.join(words[middle:])])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('plots/matrix.png')
    plt.show()


if __name__ == "__main__":
    main()
