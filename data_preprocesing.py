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

# масштабирование
# масштабирование признаков в векторе для ровного поля для тренировки алгоритма машинного обучения
data_scaler_min_max = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler_min_max = data_scaler_min_max.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaler_min_max)
# максимальное значение равно 1, остальные определяются относительно него

# нормализация. L1-нормализация применяет метод наименьших абсолютных отклонений и обеспечивает равенство 1 абсолютных значений в каждом ряду
# L2-нормализация использует метод наименьших квадратов и обеспечивает равенство 1 квадратов значений в ряду
# L1 более надежный вариант, он менее чувствителен к выбросам

data_normalazed_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalazed_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 =\n", data_normalazed_l1)
print("\nL2 =\n", data_normalazed_l2)

print()
vector_summary_l1 = 0
for vector in data_normalazed_l1[0]:
    vector_summary_l1 += vector
print(vector_summary_l1)

vector_summary_l2 = 0
for vector in data_normalazed_l2[0]:
    vector_summary_l2 += vector
print(vector_summary_l2)

