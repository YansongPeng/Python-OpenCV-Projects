# Logistic Regression to setup a perceptron module

import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    line = plt.plot(x1, x2, '-')
    plt.pause(0.0000001)
    line[0].remove()

def sigmoid(score):
    return 1/(1 + np.exp(-score))

def calc_error(line_parameters, points, y):
    m = points.shape[0]
    # Probabilities
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1/m) * (np.log(p).transpose() * y + np.log(1-p).transpose() * (1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(5000):
        p = sigmoid(points * line_parameters)
        # using alpha to do a small modifies every time
        gradient = (points.transpose() * (p - y)) * (alpha/m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
        # w1x1 + w2x2 + b = 0
        x2 = -b/w2 + x1*(-w1/w2)
        draw(x1, x2)
        print(calc_error(line_parameters, points, y))


n_pts = 100
# Get the same result everytime run the random value
np.random.seed(0)
bias = np.ones(n_pts)
random_x1 = np.random.normal(10, 2, n_pts)
random_x2 = np.random.normal(12, 2, n_pts)
random_y1 = np.random.normal(5, 2, n_pts)
random_y2 = np.random.normal(6, 2, n_pts)
top_region = np.array([random_x1, random_x2, bias]).transpose() # transpose() or T to switch raw to column
bottom_region = np.array([random_y1, random_y2, bias]).transpose()
all_points = np.vstack((top_region, bottom_region))


line_parameters = np.matrix(np.zeros(3)).transpose() # transpose() the function inorder to match the all_points matrix

#print(x1, x2)
combine_linear = all_points * line_parameters
#print(combine_linear)
probabilities = sigmoid(combine_linear)

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)


_, ax = plt.subplots(figsize = (4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'b')
# Draw the linear line
gradient_descent(line_parameters, all_points, y, 0.05)
plt.show()

#print((calc_error(line_parameters, all_points, y)))
