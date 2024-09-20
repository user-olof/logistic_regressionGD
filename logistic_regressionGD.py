import numpy as np
import iris
import decision_region
import matplotlib.pyplot as plt

class LogisticRegressionGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        rgen = np.random.RandomState(1)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            error = (y[:, 0] - output)
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            output_log1 = np.log(output, where=output > 0 ) 
            output_log2 = np.log(1-output, where=1-output > 0 ) 
            cost = (-y[:,0].dot(output_log1)) - ((1 - y[:,0]).dot(output_log2))
            self.cost_.append(cost)
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return (1 / (1 + np.exp(np.clip(-z, -250, 250))))
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

if __name__ == '__main__':
    plt.isinteractive()
    features = ['petal length (cm)', 'petal width (cm)']
    flowers = ['setosa', 'versicolor']
    X, y = iris.load_dataset_std(flowers, features)

    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000)
    lrgd.fit(X, y)

    decision_region.plot(X, y, 
                         xlabel="petal length [standardized]",
                         ylabel="petal width [standardized]",
                         classifier=lrgd)

