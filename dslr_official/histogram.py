import os
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

    house_groups = data_set.df.groupby('Hogwarts House')
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    try:
        os.makedirs('plots/histograms', exist_ok=True)
    except:
        print('Error: couldnt create plot directory', file=stderr)
        return

    for feature in data_set.num_features:
        for name, group in house_groups[feature]:
            plt.hist(group, label=name, color=colors[name], alpha=0.3)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Count')
        plt.ylabel('Grade')
        plt.title(feature)
        plt.savefig(f'plots/histograms/{feature}.png')
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
