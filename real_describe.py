import sys
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        exit()

    filename = sys.argv[1]

    file = pd.read_csv(filename)
    print(file.describe())