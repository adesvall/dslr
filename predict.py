import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

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

def predict_house(student, thetas):
    max = 0
    for house in houses:
        if predict(thetas[house], student) >= max:
            max = predict(thetas[house], student)
            best = house
    return best

#####################################################################

# Recuperation des donnees depuis result.csv
try:
    thetas = pd.read_csv('result.csv').to_dict()
    for house in houses:
        arr = []
        for val in thetas[house]:
            arr.append(thetas[house][val])
        thetas[house] = arr
    print(thetas)
    data = pd.read_csv(sys.argv[1])
except:
    print('Error: result.csv not found')
    sys.exit(1)

# Process and save
resfile = pd.DataFrame(columns=['Hogwarts House'])
for stud in data.iloc:
    resfile['Hogwarts House'][stud["Index"]] = predict_house(stud, thetas)
resfile.to_csv('result')





