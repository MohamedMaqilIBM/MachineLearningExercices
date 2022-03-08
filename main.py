import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lr = LinearRegression(normalize= True)

cov = pd.read_csv("cov.csv")

print(cov.shape)

print(cov.columns)

print(cov.groupby('Country')['Cumulative_cases', 'Cumulative_deaths'].max())

a = cov.groupby('Country')['New_cases', 'New_deaths'].sum()
print (a)

print(a.sort_values('New_deaths'))

a.sort_values('New_deaths', ascending=False, inplace=True)
print(a)

a = a.rename(columns={'New_cases':'Cases', 'New_deaths':'Deaths'})
print(a)

a['pct'] = a['Deaths']/a['Cases']*100
print(a)

a['pct'].head(20).plot.bar()
mar_data = cov.groupby('Country')

house = pd.read_csv("house.csv")
print(house.shape)
print(house)

print(y)

x_train, x_test, y_train, y_test = train_test_split(house.loyer, house.surface, test_size = 0.2)

model = LinearRegression()

model.fit(np.array(x_train).reshape(-1,1), y_train)

predictions = model.predict(np.array(x_test).reshape(-1,1))

plt.scatter()
plt.hist(predictions - y, bins =50)

# preds = model.predict(np.array(x_test).reshape(-1,1))