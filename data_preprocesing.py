import numpy as np
from sklearn import preprocessing

# определение выборки данных
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

# бинаризация
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print(data_binarized)

# все значения выше 2.1 принудительно устанавливаются равными 1

# исключение среднего
print("\nBefore:")
print("Mean =", input_data.mean(axis=0))  # среднее значение
print("Standard deviation =", input_data.std(axis=0))  # стандартное отклонение

data_scaled = preprocessing.scale(input_data)
print("\nAfter:")
print("Mean =", data_scaled.mean(axis=0))
print("Standard deviation =", data_scaled.mean(axis=0))
