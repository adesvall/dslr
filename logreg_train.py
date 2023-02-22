import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("datasets/dataset_train.csv")

df_train = df.sample(frac=0.5, random_state=0)
df_test = df.drop(df_train.index)

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
#    "Defense Against the Dark Arts", linked to Astronomy
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
# on va faire 4 theta, un pour chaque maison

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
        sum += y * math.log(predict(theta, student))
        sum += (1 - y) * math.log(1 - predict(theta, student))
    return sum / len(df)

def gradient(theta, house):
    grad = [0] * (len(features) + 1)
    for student in df.iloc:
        y = (student["Hogwarts House"] == house)
        for i in range(len(features)):
            if not pd.isna(student[features[i]]):
                grad[i + 1] += (predict(theta, student) - y) * student[features[i]]
    for i in range(len(grad)):
        grad[i] /= len(df)
    return grad

def gradient_descent(theta, house, alpha):
    grad = gradient(theta, house)
    for i in range(len(theta)):
        theta[i] -= alpha * grad[i]
    return theta

def train(house):
    theta = [0] + [0] * len(features)
    for i in range(30):
        theta = gradient_descent(theta, house, 0.00001)
        if i % 3 == 0:
            print("iter :", i, "accuracy :", onevsall_accuracy(theta, house), end='\r')
    return theta

def onevsall_accuracy(theta, house):
    correct = 0
    for student in df.iloc:
        if (predict(theta, student) > 0.5) == (student["Hogwarts House"] == house):
            correct += 1
    return correct / len(df)

thetas = {}
for house in houses:
    thetas[house] = train(house)
    print(house, "accuracy :", onevsall_accuracy(thetas[house], house))

def accuracy(thetas):
    correct = 0
    for student in df.iloc:
        max = 0
        for house in houses:
            if predict(thetas[house], student) > max:
                max = predict(thetas[house], student)
                best = house
        if best == student["Hogwarts House"]:
            correct += 1
    return correct / len(df)

print("accuracy :", accuracy(thetas))

