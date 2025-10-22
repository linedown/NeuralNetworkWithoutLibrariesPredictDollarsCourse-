import math
import numpy as np
import pandas as pd

# Класс нейросети-перцептрона для прогнозирования
class ForecastNet:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        # Инициализация весов и смещений
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bias2 = np.zeros(hidden_dim)
        self.weights3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bias3 = np.zeros(output_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, x):
        # Первый слой
        self.layer1_input = x @ self.weights1 + self.bias1
        self.layer1_output = self.relu(self.layer1_input)
        # Второй слой
        self.layer2_input = self.layer1_output @ self.weights2 + self.bias2
        self.layer2_output = self.relu(self.layer2_input)
        # Выходной слой
        self.output = self.layer2_output @ self.weights3 + self.bias3

        return self.output

    def backward(self, x, y, output):
        # Вычисляем градиенты для выходного слоя
        output_error = output - y
        # Линейная активация, производная = 1
        output_delta = output_error
        # Градиенты для третьего слоя
        weights3_gradient = self.layer2_output.T @ output_delta
        bias3_gradient = np.sum(output_delta, axis=0)
        # Градиенты для второго слоя
        layer2_error = output_delta @ self.weights3.T
        layer2_delta = layer2_error * self.relu_derivative(self.layer2_input)
        # Градиенты для второго слоя
        weights2_gradient = self.layer1_output.T @ layer2_delta
        bias2_gradient = np.sum(layer2_delta, axis=0)
        # Градиенты для первого слоя
        layer1_error = layer2_delta @ self.weights2.T
        layer1_delta = layer1_error * self.relu_derivative(self.layer1_input)
        # Градиенты для первого слоя
        weights1_gradient = x.T @ layer1_delta
        bias1_gradient = np.sum(layer1_delta, axis=0)
        # Обновляем веса и смещения
        self.weights3 -= self.learning_rate * weights3_gradient
        self.bias3 -= self.learning_rate * bias3_gradient
        self.weights2 -= self.learning_rate * weights2_gradient
        self.bias2 -= self.learning_rate * bias2_gradient
        self.weights1 -= self.learning_rate * weights1_gradient
        self.bias1 -= self.learning_rate * bias1_gradient
    
    def train(self, x, y):
        output = self.forward(x)
        self.backward(x, y, output)
    
    def mse_loss(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)
    
# Размер скользящего окна
wnd_size = 20
# Число нейронов во внутреннем слое
inner_cnt = 128
# Максимальное расхождение при обучении
eps = 0.05
# Число элементов в прогнозе
forecast_len = 100
# Число эпох (шагов в цикле при обучении)
epochs = 2000
# Получение данных из excel-файла в формате DataFrame
data = pd.read_excel(r'datasetUSD.xlsx', index_col='Date', sheet_name='Worksheet',
                     names=['Date','Value'])
# Формирование датасета
dataset = np.asarray(data['Value'])
# Нормализация данных
data_mean = np.mean(dataset)
data_std = np.std(dataset)
norm_data = (dataset - data_mean) / data_std
# Массив дат
dates = np.asarray(data.index)
# Размер датасета
ds_len = len(dataset)
# Создание модели нейросети
model = ForecastNet(wnd_size, inner_cnt, 1)

# Обучение нейросети
for epoch in range(epochs):
    for i in range(wnd_size, ds_len):
        # Reshape для соответствия входной размерности
        inp = norm_data[i - wnd_size : i].reshape(1, -1)
        target = norm_data[i]
        # Передаем numpy array
        model.train(inp, np.array([[target]]))

# Массив для прогнозирования
forecast_arr = norm_data.copy()
# Массив дат для прогнозирования
forecast_dates = pd.date_range(dates[-1], inclusive="right", periods=forecast_len + 1)

# Прогноз
for i in range(ds_len, ds_len + forecast_len):
    inp = forecast_arr[i - wnd_size : i].reshape(1, -1)
    outp = model.forward(inp)[0][0]
    forecast_arr = np.append(forecast_arr, outp)

# Денормализация
forecast_arr = forecast_arr * data_std + data_mean

forecast = pd.DataFrame({
    'Date': forecast_dates,
    'USD': forecast_arr[ds_len:]
})

forecast.to_excel('usd_forecast_without.xlsx', index = False)
