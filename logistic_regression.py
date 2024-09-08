import numpy as np
import data
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

class SGDLogisticRegressor:
    def __init__(self, lr, num_iter):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.num_iter=num_iter

    def sigmoid_function(self, x):
        return 1/(1+np.exp(-x))

    def gradient_desc_weight(self, x, y, y_pred):
        return x * (y_pred - y)

    def gradient_desc_bias(self, y, y_pred):
        return y_pred - y

    def linear(self,x):
        value_mult = 0
        for i in range(len(self.weights)):
            value_mult += self.weights[i] * x[i]
        return value_mult + self.bias

    def calc_loss(self, y, y_pred):
        #cross entropy loss
        loss = -y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return loss

    #generates bias and weights that fits to the values
    def train(self, X, y):

        #initialisation of weights and bias
        self.weights = [0] * len(X[0])
        self.bias = 0
        loss_array = []

        for n in range(self.num_iter):
            # stochastic gradient descend evaluates after every step
            for i in range(len(X)):
                x_calc = self.linear(X[i])
                y_pred = self.sigmoid_function(x_calc)

                # gradient calculating
                gradient_weight = self.gradient_desc_weight(X[i], y[i], y_pred)
                gradient_bias = self.gradient_desc_bias(y[i], y_pred)

                # updating weight and bias
                self.weights -= self.lr * gradient_weight
                self.bias -= self.lr * gradient_bias
            loss = self.calc_loss(y[i], y_pred)
            loss_array.append(loss)

        return loss_array

    def test(self, X):
        predicted_values = []
        for s in range(len(X)):
            x_calc = self.linear(X[s])
            y_pred = self.sigmoid_function(x_calc)
            if y_pred > 0.5:
                predicted_values.append(1)
            else:
                predicted_values.append(0)
        return predicted_values


# Example usage
if __name__ == "__main__":

    # generate the datasets
    X_train, X_test, y_train, y_test, X, y = data.create_data()

    model = SGDLogisticRegressor(lr=0.001, num_iter=100)

    loss = model.train(X_train, y_train)

    print("Accuracy train data", accuracy_score(y_train, model.test(X_train)))

    predictions = model.test(X_test)

    #plot the loss
    epochs = [i for i in range(model.num_iter)]
    plt.plot(epochs,loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # plot the data
    plt.figure()
    plt.subplots_adjust()
    plt.subplot(111)
    plt.xlabel('liveness')
    plt.ylabel('loudness')
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    plt.show()

    # plot the linear line seperating the data
    liveness_values = np.linspace(0, 1, 50)
    # equation for getting loudness values with the weights
    loudness_values = - (model.bias + model.weights[0] * liveness_values) / model.weights[1]
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    plt.plot(liveness_values, loudness_values, linestyle='--', color='green')
    plt.xlabel('liveness')
    plt.ylabel('loudness')
    plt.legend()
    plt.show()

    # evaluate the logistic regression model
    acc = accuracy_score(y_test, predictions)
    print("Accuracy test data", acc)
    confusion_m = confusion_matrix(y_test, predictions)
    print(confusion_m)
