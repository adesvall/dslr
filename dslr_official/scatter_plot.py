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

    x_feature = 'Defense Against the Dark Arts'
    y_feature = 'Astronomy'

    house_groups = data_set.df.groupby('Hogwarts House')
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    try:
        os.makedirs('plots', exist_ok=True)
    except:
        print('Error: couldnt create plot directory', file=stderr)
        return

    try:
        plt.figure(figsize=[10, 10])
        for name, group in house_groups:
            plt.scatter(x=group[x_feature], y=group[y_feature], color=colors[name], alpha=0.5, label=name)
        plt.legend()
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.savefig('plots/scatter.png')
        plt.show()
        plt.close()
    except:
        print(f'Error: couldnt scatter "{x_feature}" against "{y_feature}"', file=stderr)
        return


if __name__ == "__main__":
    main()
