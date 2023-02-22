import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/dataset_train.csv")
pd.plotting.scatter_matrix(df)
plt.show()