import numpy as np
import pandas as pd
import io
from sklearn.neural_network import MLPClassifier
from sklearn.utils import column_or_1d

y = pd.read_excel(r'/Users/mohamedmaqil/PycharmProjects/SchoolProject/files/OUTPUT.xlsx')
print(y)
y = column_or_1d(y, warn=True)

x = pd.read_excel(r'/Users/mohamedmaqil/PycharmProjects/SchoolProject/files/DATA.xlsx')
print(x)

Xnew = pd.read_excel(r'/Users/mohamedmaqil/PycharmProjects/SchoolProject/files/DATAtest.xlsx')
print(Xnew)

clfrna = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(6, 3), random_state=1, max_iter=2000)
clfrna.fit(x, y)