import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

df = pd.read_excel('datasetUSD.xlsx', parse_dates=['Дата'])
df_sorted = df.sort_values(by='Дата')

df_predict = pd.read_excel('usd_forecast_without.xlsx', parse_dates=['Date'])
df_predict_sorted = df_predict.sort_values(by='Date')

# Настройка фигуры
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(df_sorted['Дата'], df_sorted['Курс доллара'], color='tab:blue', linewidth=0.8)
ax.plot(df_predict_sorted['Date'], df_predict_sorted['USD'], color='tab:orange', linewidth=0.8)

# Форматирование оси X: метки по годам
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

# Поворот и отступ меток
plt.xticks(rotation=45)

ax.set_title('Курс доллара по датам (2009-2026)')
ax.set_xlabel('Год')
ax.set_ylabel('Курс доллара')
ax.grid(alpha=0.3)

# Ограничим ось X по крайним датам в данных
ax.set_xlim([df_sorted['Дата'].min(), df_predict_sorted['Date'].max()])

# Сохраняем файл
output_filename = 'dollar_course200.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

# Печатаем информацию для пользователя
print(f"Сохранён файл: {output_filename}")
