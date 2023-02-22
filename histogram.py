import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/dataset_train.csv")
df_num = df.select_dtypes(include='number')

houses = [
    "Slytherin",
    "Ravenclaw",
    "Gryffindor",
    "Hufflepuff",
]
colors = [
    "green",
    "blue",
    "red",
    "yellow",
]
effectifs = {}
for house in houses:
    effectifs[house] = df[df["Hogwarts House"] == house].shape[0]
print(effectifs)

plt.style.use('dark_background')
for course in df_num:
    # plt.hist(df[df["Hogwarts House"] == "Slytherin"][course], alpha=0.5)
    # plt.hist(df[df["Hogwarts House"] == "Ravenclaw"][course], alpha=0.5)
    # plt.hist(df[df["Hogwarts House"] == "Gryffindor"][course], alpha=0.5)
    # plt.hist(df[df["Hogwarts House"] == "Hufflepuff"][course], alpha=0.5)
    plt.hist([df[df["Hogwarts House"] == house][course] for house in houses], density=True, label=houses, color=colors)
    plt.legend(prop={'size': 10})
    plt.title(course)
    plt.show()

exit()
