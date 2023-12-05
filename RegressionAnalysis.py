import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np

listHigh = [153.3,164.9,168.1,151.5,157.8,156.7,161.1]
listWeight =[45.5,56.0,55.0,52.8,55.6,50.8,56.4]
listPrediction = []
listError = []

p = 1
q = 1
SumError = 0

for i in range(len(listHigh)):
    listPrediction.append(p * listHigh[i] + q)
    listError.append((listWeight[i] - listPrediction[i]) ** 2)
    SumError += listError[i]

print("---p=1,q=1---")
print(listPrediction)
print(listError)
print(SumError)

x = np.linspace(0, 180, 200)
y = p*x**1 + q*x**0

plt.axis([0, 180, 0, 200])
plt.plot(listHigh, listWeight, 'o')
plt.plot(y)
plt.show()

print("---p,q最適化---")
df = pd.DataFrame({'High': listHigh, 'Weight': listWeight})

x = df[['High']]
y = df[['Weight']]

model_lr = LinearRegression()
model_lr.fit(x , y)

plt.axis([130, 180, 40, 60])
plt.plot(x, y, 'o')
plt.plot(x, model_lr.predict(x), linestyle='solid')
plt.show()

print("p = %.5f  %.5f" % (model_lr.coef_, model_lr.intercept_))
