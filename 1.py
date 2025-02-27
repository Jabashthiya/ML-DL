from sklearn.linear_model import LinearRegression
import numpy as np
x = np.array([[1], [2], [3], [4], [5]])
# Labels
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(x, y)
test_data = np.array([[6]])
prediction = model.predict(test_data)
print("Prediction for input {}: {}".format(test_data.flatten(),
prediction[0]))
