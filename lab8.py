import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Класс нейросети-перцептрона для прогнозирования
class ForecastNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ForecastNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

# Размер скользящего окна
wnd_size = 20
# Число нейронов во внутреннем слое
inner_cnt = 128
# Максимальное расхождение при обучении
eps = 0.05
# Число элементов в прогнозе
forecast_len = 100

# Получение данных из excel-файла в формате DataFrame 
data = pd.read_excel(r'datasetUSD.xlsx',
                     index_col='Date', sheet_name='Worksheet',
                     names=['Date','Value'])

# Формирование датасета
dataset = torch.tensor(np.asarray(data['Value']), dtype=torch.float32)
# Массив дат
dates = np.asarray(data.index)
# Размер датасета
ds_len = len(dataset)

# Создание модели нейросети
model = ForecastNet(wnd_size, inner_cnt, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Обучение нейросети
for i in range(wnd_size, ds_len):
    while True:
        optimizer.zero_grad()
        inp = dataset[i - wnd_size : i]
        outp = model(inp)
        if abs(outp - dataset[i:i+1]) < eps:
            break
        loss = loss_fn(outp, dataset[i:i+1])
        loss.backward()
        optimizer.step()

# Массив для прогнозирования
forecast_arr = dataset.clone()
# Массив дат для прогнозирования
forecast_dates = pd.date_range(dates[-1], inclusive="right",
                               periods=forecast_len+1)

# Прогноз
for i in range(ds_len, ds_len + forecast_len):
    inp = forecast_arr[i - wnd_size : i]
    outp = model(inp)
    forecast_arr = torch.cat((forecast_arr,outp))

forecast = pd.DataFrame({
    'Date': forecast_dates,
    'USD': forecast_arr[ds_len:].detach().numpy()
})

forecast.to_excel('usd_forecast.xlsx', index=False)
