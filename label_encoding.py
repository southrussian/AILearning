import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'black']  # метки входных данных
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels) # создание объекта кодирования меток и обучение его

print("Label mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, "-->", i)  # вывод отображения слов на числа

