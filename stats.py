import pandas as pd

#create DataFrame
df = pd.DataFrame({'points': [1, 1, 2, 3.5, 4, 4, 4, 5, 5, 6.5, 7, 7.4, 8, 13, 14.2],
                   'assists': [5, 7, 7, 9, 12, 9, 9, 4, 6, 8, 8, 9, 3, 2, 6],
                   'rebounds': [11, 8, 10, 6, 6, 5, 9, 12, 6, 6, 7, 8, 7, 9, 15]})

df.head()
print(df)

#calculate mean of 'points'
df['points'].mean()

#calculate median of 'points'
df['points'].median()

#calculate standard deviation of 'points'
df['points'].std()

#create frequency table for 'points'
df['points'].value_counts()

