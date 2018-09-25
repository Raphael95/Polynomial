import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
from sklearn import linear_model


class LinearRegression:

    def  __init__(self):
        self.num_of_samples = 0   # number of samples
        self.num_of_features = 0   # number of features
        self.weight = 0

    def get_weight(self):
        return self.weight

    # Batch Gradient Descent
    def fit_batch_gradient(self, x, y):
        self.num_of_samples = np.shape(x)[0]
        self.num_of_features = np.shape(x)[1]

        alpha = 0.1
        error = np.asarray([np.zeros(self.num_of_features)]).transpose()
        threshold = 0.000000001
        count = 0
        max_loop = 10000
        np.random.rand(0)
        #self.weight = np.asarray([np.random.rand(self.num_of_features)]).transpose()  # define the weight vector
        self.weight = np.asarray([np.linspace(0, 1, self.num_of_features)]).transpose()

        # 此处的循环是为了防止权值误差无法收敛时设置的最大循环次数
        while count < max_loop:
            count += 1
            print(count)
            sum_diff = np.asarray([np.zeros(self.num_of_features)]).transpose()

            # 循环遍历样本，不断更新权值weight
            for i in range(self.num_of_samples):
                diff = (np.dot(x[i], self.weight) - y[i]) * (np.asarray([x[i]]).transpose())
                sum_diff += diff
            self.weight -= alpha * sum_diff
            print(self.weight)

            # # 判断推出循环的条件，若权值误差开始convergence，则停止训练更新weight
            # if np.linalg.norm(self.weight - error) < threshold:
            #     break
            # else:
            #     error = self.weight

    # Stochastic Gradient Descent
    def fit_stochastic_gradient(self, x, y):
        self.num_of_samples = np.shape(x)[0]
        self.num_of_features = np.shape(x)[1]

        alpha = 0.1
        error = np.asarray([np.zeros(self.num_of_features)]).transpose()
        threshold = 0.00001
        count = 0
        max_loop = 10000
        np.random.rand(0)
        # self.weight = np.asarray([np.random.rand(self.num_of_features)]).transpose()  # define the weight vector
        self.weight = np.asarray([np.linspace(0, 1, self.num_of_features)]).transpose()

        # 此处的循环是为了防止权值误差无法收敛时设置的最大循环次数
        while count < max_loop:
            count += 1
            print(count)

            # 循环遍历样本，不断更新权值weight
            for i in range(self.num_of_samples):
                diff = (np.dot(x[i], self.weight) - y[i]) * (np.asarray([x[i]]).transpose())
                self.weight -= alpha * diff
            print(self.weight)


    def predict(self, x):
        y = np.dot(x, self.weight)
        return y




def generate_data(num, std):
    x = np.linspace(0, 1, num)
    y = func_sin(x) + np.random.normal(scale = std, size = x.shape)
    return x, y

def func_sin(x):
    return np.sin(2 * np.pi * x)


x_train, y_train = generate_data(10, 0.25) # generate train_data x,y with noise
x_test = np.linspace(0, 1, 100)  # generate test_data x,y without noise
y_test = func_sin(x_test)
print(x_train, y_train)
print(x_test, y_test)

plt.scatter(x_train, y_train, facecolor= None, edgecolors='b', s = 50, label='training_data')
plt.plot(x_test, y_test, c = 'g', label='$\sin(2\pi x)$')
plt.legend()
plt.show()


def transform_data(x, degree):
    if x.ndim == 1:
        x = x[:, None]
    x_t = x.transpose()
    features = [np.ones(len(x))]
    for d in range(1, degree+1):
        for item in itertools.combinations_with_replacement(x_t, d):
            features.append(functools.reduce(lambda x, y : x * y, item))
    return np.asarray(features).transpose()

def trans_data(x, degree):
    x_1 = [i for i in x]
    x_2 = [i * i for i in x]
    x_3 = [i * i * i for i in x]
    x_4 = [i * i * i * i for i in x]
    feature = [x]
    feature.append(x_1)
    feature.append(x_2)
    feature.append(x_3)
    feature.append(x_4)
    feature = np.asarray(feature)
    return feature.transpose()



degree = 4
X_train = trans_data(x_train, degree)
X_test = trans_data(x_test, degree)



model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y1 = model.predict(X_test)


linear_regression = LinearRegression()
linear_regression2 = LinearRegression()
linear_regression.fit_batch_gradient(X_train, y_train)
linear_regression2.fit_stochastic_gradient(X_train, y_train)
y2 = linear_regression.predict(X_test)
y3 = linear_regression2.predict(X_test)


plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y1, c="r", label="sys fitting")
plt.plot(x_test, y2, c="y", label="batch gradient")
plt.plot(x_test, y3, c="b", label="stochastic gradient")
plt.ylim(-1.5, 1.5)
plt.annotate("M={}".format(degree), xy=(-0.15, 1))
plt.legend(bbox_to_anchor=(0.98, 0.96))
plt.show()
