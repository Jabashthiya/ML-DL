from sklearn.linear_model import LogisticRegression
import numpy as np
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])
# Create and train the model
model = LogisticRegression()
model.fit(x, y)
test_data = np.array([[2.5], [3.5]])
predictions = model.predict(test_data)
print("Predictions for inputs {}: {}".format(test_data.flatten(),
predictions))
