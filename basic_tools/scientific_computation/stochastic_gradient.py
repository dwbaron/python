import random
import numpy as np


# shuffle the data by indexes
def in_random_order(data):
    indexes = list(range(len(data)))
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


# evaluate the parameters given the target cost function
# support vectors as input
def minimize_stochastic(target_f, gradient_f, x, y, theta, alpha_0=0.01):
    data = list(zip(x, y))
    theta = theta
    alpha = alpha_0
    min_theta, min_value = None, float('inf')
    iter_with_no_improvement = 0

    while iter_with_no_improvement < 100:
        # if reach the tolerance return
        if min_value <= 0.00001:
            return min_theta
        # sum of squared errors
        value = sum(target_f(x, y, theta))
        print(value, min_value)
        # if found a minimum reset the step size
        if value <= min_value:
            min_theta, min_value = theta, value
            iter_with_no_improvement = 0
            alpha = alpha_0

        else:
            # if not shrink the step size
            iter_with_no_improvement += 1
            alpha *= .9

        # adjust the theta at each point by random order
        gradients = []
        for theta_i, point in zip(theta, in_random_order(data)):
            gradients.append(gradient_f(point[0], point[1], theta_i))

        theta = theta - [alpha * gradient_i for gradient_i in gradients]

    return min_theta


if __name__ == '__main__':

    import numpy as np
    # given x and y sequences
    # our goal is to find the theta that minimize the target cost function
    x_s = np.linspace(-2, 2, 10)
    # the scalar 2 is the target value of theta
    y_s = 2 * x_s**2

    # initial values of theta:
    # [3, 3, ... , 3]
    theta = np.ones(10) * 1

    # target cost function: squared error
    def t(x, y, theta):
        return (y - theta * x**2)**2

    # the derivative on theta
    def dt(x, y, theta):
        return 2 * x**4 * theta - 2 * y * x**2

    min_theta = minimize_stochastic(t, dt, x_s, y_s, theta)
    # the result should be close to [2, 2, ... , 2]
    print('result:', min_theta)