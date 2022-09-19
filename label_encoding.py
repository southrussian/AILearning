import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'black']  # метки входных данных
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels) # создание объекта кодирования меток и обучение его

print("Label mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, "-->", i)  # вывод отображения слов на числа

# преобразование меток с помощью кодировщика
test_labels = ['red', 'black', 'yellow']
encoded_values = encoder.transform(test_labels)
print('\nLabels:\n', test_labels)
print('\nEncoded values:\n', encoded_values)

# декодирование случайного набора чисел
values = [5, 2, 0, 3]
decoded_list = encoder.inverse_transform(values)
print('\nValues:\n', values)
print('\nDecoded list:\n', decoded_list)



