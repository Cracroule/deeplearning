import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.datasets
from deeplearning.plot import plot_decision_boundary
from deeplearning.helpers import L_layer_model, L_model_forward
import numpy as np
import matplotlib

np.random.seed(1)
X, y = sklearn.datasets.make_moons(400, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Train the logistic rgeression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)

total_sample_size = X.shape[0]
train_ratio = 0.8
train_size = int(X.shape[0] * train_ratio)
test_size = total_sample_size - train_size
X_train = X[:train_size, :]
y_train = y[:train_size]
X_test = X[train_size:, :]
y_test = y[train_size:]

# hyperparameters choices
nb_of_hidden_layouts = 2
nb_of_units_per_hidden_layouts = 8

hidden_lay_dims = [nb_of_units_per_hidden_layouts] * nb_of_hidden_layouts
layers_dims = (X_train.shape[1], *hidden_lay_dims, 1)




# Reshape the training and test examples
Y_train = y_train.reshape(y_train.shape[0], 1).T
X_train = X_train.reshape(X_train.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
Y_test = y_test.reshape(y_test.shape[0], 1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
print('X_train shape', X_train.shape)
print('Y_train shape', Y_train.shape)
print('X_test shape', X_test.shape)
print('Y_test shape', Y_test.shape)


#plot_decision_boundary(X, y, lambda x: clf.predict(x))
#plot_decision_boundary(X_train.T, y_train.reshape(y_train.shape[0], ).T, lambda x: clf.predict(x))
# plt.title("Logistic Regression")
# plt.show()


opti_params = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.01, num_iterations=30000, lambda_reg=0.4,
                            print_cost=True)

y_train_hat, cache = L_model_forward(X_train, opti_params)
y_test_hat, cache = L_model_forward(X_test, opti_params)
train_predictions = y_train_hat > 0.5
test_predictions = y_test_hat > 0.5
print("train accuracy:", np.sum(train_predictions == Y_train) / train_size)
print("test accuracy:", np.sum(test_predictions == Y_test) / test_size)


# returns a prediction with chosen input
# care, works with inverted x and y compared to deeplearning algo
def my_predict(x):
    y, cache = L_model_forward(x.T, opti_params)
    predictions = y > 0.5
    return predictions.T

plot_decision_boundary(X_train.T, y_train.reshape(y_train.shape[0], ).T, my_predict)
plt.show()

