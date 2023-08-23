import sys
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class DataSet:
    def __init__(self, filepath: str):
        try:
            self.df = pd.read_csv(filepath)
            self.df = self.df.dropna(how='all', axis=1)

            self.num_features = [feature for feature in self.df
                                if is_numeric_dtype(self.df[feature]) and feature != 'Index']

            self.descriptions = {
                'Count': self.count,
                'Mean': self.mean,
                'Std': self.std,
                'Min': self.min,
                '5%': self.five,
                '25%': self.first_quartile,
                '50%': self.median,
                '75%': self.third_quartile,
                '95%': self.ninetyfive,
                'Max': self.max,
                'Missing %': self.missing,
                'Negative %': self.negative,
                'Distinct %': self.distinct,
                'Skewness': self.skewness,
            }
        except:
            raise ValueError(f"Error: invalid csv file '{filepath}'")


    def count(self, df: pd.DataFrame) -> int:
        try:
            values = np.array(df)
            values = values[~np.isnan(values)]
            return len(values)
        except:
            return 0


    def mean(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            values = values[~np.isnan(values)]
            count = len(values)
            values /= count

            return values.sum()
        except:
            return 0


    def std(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            values = values[~np.isnan(values)]
            count = len(values)
            if count == 0:
                return 0

            mean = self.mean(df)
            total = (values - mean) ** 2
            return math.sqrt(total.sum() / (count - 1))
        except:
            return 0
    
    def skewness(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            values = values[~np.isnan(values)]
            count = len(values)
            if count == 0:
                return 0

            mean = self.mean(df)
            std = self.std(df)
            total = ((values - mean) / std) ** 3
            return total.sum() * count / (count - 1) / (count - 2)
        except:
            return 0


    def min(self, df: pd.DataFrame) -> float:
        try:
            tmp_min = None

            for n in df:
                if tmp_min is None or n < tmp_min:
                    tmp_min = n
            return tmp_min
        except:
            return 0


    def max(self, df: pd.DataFrame) -> float:
        try:
            tmp_max = None

            for n in df:
                if tmp_max is None or n > tmp_max:
                    tmp_max = n
            return tmp_max
        except:
            return 0

    def percentile(self, df, p):
        try:
            values = np.array(df)
            values = values[~np.isnan(df)]
            n = len(values) - 1

            p = n * p / 100
            values.sort()
            return values[int(p)] * (1 - p % 1) + values[int(p) + 1] * (p % 1)
        except:
            return None

    def median(self, df: pd.DataFrame) -> float:
        return self.percentile(df, 50)

    def first_quartile(self, df: pd.DataFrame) -> float:
        return self.percentile(df, 25)

    def third_quartile(self, df: pd.DataFrame) -> float:
        return self.percentile(df, 75)
    
    def five(self, df: pd.DataFrame) -> float:
        return self.percentile(df, 5)
    
    def ninetyfive(self, df: pd.DataFrame) -> float:
        return self.percentile(df, 95)


    def missing(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            total = len(values)
            count = len(values[np.isnan(values)])

            return (count / total) * 100
        except:
            return 0


    def negative(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            total = len(values)
            count = len(values[values < 0])

            return (count / total) * 100
        except:
            return 0


    def distinct(self, df: pd.DataFrame) -> float:
        try:
            values = np.array(df)
            total = len(values)
            count = len(np.unique(values))

            return (count / total) * 100
        except:
            return 0


    def describe(self) -> pd.DataFrame | None:
        try:
            indexes = self.descriptions.keys()
            columns = self.num_features

            values = [
                [method(self.df[feature]) for feature in self.num_features]
                for method in self.descriptions.values()
            ]

            return pd.DataFrame(values, index=indexes, columns=columns)
        except:
            return None


    @staticmethod
    def split(x, y, ratio):
        try:
            total = np.concatenate((x, y), axis=1)
            np.random.shuffle(total)

            limit = math.floor(len(total) * ratio)

            train, test = total[:limit], total[limit:]

            train_x, test_x = train[:, :-1], test[:, :-1]
            train_y, test_y = train[:, -1], test[:, -1]

            if train_x.shape[1] == 1:
                train_x = train_x.ravel()
                test_x = test_x.ravel()

            return (train_x, test_x, train_y, test_y)
        except:
            return None
