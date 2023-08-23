import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

houses = [
    "Slytherin",
    "Ravenclaw",
    "Gryffindor",
    "Hufflepuff",
]

features = [
#    "Arithmancy", homogeneous
    "Astronomy",
    "Herbology",
#    "Defense Against the Dark Arts", similar to Astronomy
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
#    "Care of Magical Creatures", homogeneous
    "Charms",
    "Flying",
]
# theta = [b, poids1, poids2, ...]
# on va faire 4 theta, un pour chaque maison (one vs all)

######################################################################


def sigmoid(x):
    try:
        res = 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0
    return res

def predict(theta, student):
    sum = theta[0]
    for i in range(len(features)):
        if not pd.isna(student[features[i]]):
            sum += theta[i + 1] * student[features[i]]
    return sigmoid(sum)

def loss(theta, house):
    sum = 0
    for student in df.iloc:
        y = (student["Hogwarts House"] == house)
        sum -= y * math.log(predict(theta, student))
        sum -= (1 - y) * math.log(1 - predict(theta, student))
    return sum / len(df)


# pred(stud) = sigm(b + sum(theta[i] * stud[feature[i]]))
# y(stud) = (stud[house] == HOUSE)
# logloss = sum(     -y * log( pred(stud) ) - (1 - y) * log(1 - pred(stud))       for stud in df)
# dpred / dtheta[i] = stud[i] * pred() * (1 - pred)
# grad[i] = dlogloss / dtheta[i] = sum(   -y * stud[i] *( 1 - pred(stud)) - (1 - y) * -stud[i] * pred    for stud in df)
#                                = sum(   stud[i] * ( y * pred[stud] - y + pred[stud] - y * pred)    for stud in df)
#                                = sum(   stud[i] * (pred[stud] - y)    for stud in df)
# dpred / db = pred * (1 - pred)
# dlogloss / db = = sum(   (pred[stud] - y)    for stud in df)


def gradient(theta, house):
    grad = np.ndarray(len(features) + 1)
    for student in df_train.iloc:
        y = (student["Hogwarts House"] == house)
        grad[0] += (predict(theta, student) - y)
        for i in range(len(features)):
            if not pd.isna(student[features[i]]):
                grad[i + 1] += (predict(theta, student) - y) * student[features[i]]
    return grad / len(df_train)

def train(house):
    theta = np.ndarray(len(features) + 1)
    for i in range(10):
        print("iter :", i, "accuracy :", onevsall_accuracy(theta, house, df_test), end='\r')
        theta -= 0.001 * gradient(theta, house)
    return theta

def onevsall_accuracy(theta, house, data):
    correct = 0
    for student in data.iloc:
        if (predict(theta, student) > 0.5) == (student["Hogwarts House"] == house):
            correct += 1
    return correct / len(data)

def accuracy(thetas, data):
    correct = 0
    for student in data.iloc:
        max = 0
        for house in houses:
            if predict(thetas[house], student) >= max:
                max = predict(thetas[house], student)
                best = house
        if best == student["Hogwarts House"]:
            correct += 1
    return correct / len(data)

##################################################################################

df = pd.read_csv("datasets/dataset_train.csv")

# normalize data
for column in features:
    df[column] = (df[column] - df[column].mean()) / df[column].std()


df_train = df.sample(frac=0.2)
df_test = df.drop(df_train.index)

thetas = {}
for house in houses:
    print(house, ':')
    thetas[house] = train(house)
    print("\ttrainset accuracy :", onevsall_accuracy(thetas[house], house, df_train))
    print("\ttestset accuracy :", onevsall_accuracy(thetas[house], house, df_test))


print("trainset accuracy :", accuracy(thetas, df_train))
print("testset accuracy :", accuracy(thetas, df_test))

# sauvegarde des donnees dans result.csv
resfile = pd.DataFrame()
resfile.to_csv('thetas.csv', index=False)
