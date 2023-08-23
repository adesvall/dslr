from sys import argv, stderr
from data_set import DataSet


def main():
    if len(argv) != 2:
        print(f'Usage: python {argv[0]} path/to/dataset')
        return

    try:
        data_set = DataSet(argv[1])
    except Exception as e:
        print(e, file=stderr)
        return

    description = data_set.describe()
    if description is None:
        print(f'Error: couldnt describe file {argv[1]}\n'
              + 'Make sure it containts numerical features', file=stderr)

    print(description)


if __name__ == "__main__":
    main()

