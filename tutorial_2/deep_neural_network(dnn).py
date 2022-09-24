
import tensorflow as tk
import numpy as np
import pandas as pd

df = pd.read_csv('D:/anaconda/keras tutorial/tutorial_2/HousingPrices.csv')

df.head()

X = df.drop(columns=['SalePrice'])
Y = df[['SalePrice']]

model=tk.keras.Sequential()

model.add(tk.keras.layers.Dense(8, activation='relu', input_shape=(8,)))
model.add(tk.keras.layers.Dense(8, activation='relu'))
model.add(tk.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=30, callbacks=[tk.keras.callbacks.EarlyStopping(patience=5)])

test_data = np.array([2003,	854,	1710,	2,	1,	3,	8,	2008])
print(model.predict(test_data.reshape(1,8)))

model.save("D:/anaconda/keras tutorial/ML_housing_price_predict.h5")

model = tk.keras.models.load_model('D:/anaconda/keras tutorial/ML_housing_price_predict.h5')

test_data = np.array([2003,	854,	1710,	2,	1,	3,	8,	2008])
print(model.predict(test_data.reshape(1,8)))

