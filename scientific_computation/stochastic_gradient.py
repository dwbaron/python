# shuffle the data by indexes
def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


# evaluate the parameters given the target function
# usually the target function is a cost function
# support the vector as input
def minimize_stochastic(target_f, gradient_f, x, y, theta, alpha=0.01):
    data = list(zip(x, y))
    theta = theta
    alpha = alpha
    min_theta, min_value = None, float('inf')
    iter_with_no_improvement = 0

    while iter_with_no_improvement < 100:
        # error value minimum is 0
        value = abs(np.sum(target_f(x, y, theta)))
        print(value, min_value)
        # if found a minimum extend the step size
        if value < min_value:
            min_theta, min_value = theta, value
            iter_with_no_improvement = 0
            alpha *= 1.1

        else:
            # if not shrink the step size
            iter_with_no_improvement += 1
            alpha *= .9

        # adjust the theta at each point by random order
        gradients = []
        for x_i, y_i in in_random_order(data):
            gradients.append(gradient_f(x_i, y_i, theta))

        theta = [theta_i - alpha * gradient for theta_i, gradient
                 in zip(theta, gradients)]

    return min_theta


if __name__ == '__main__':

    import numpy as np
    x = np.linspace(-1, .5 * np.pi, 10)
    y = 2 * np.sin(x)
    # initial values of theta
    theta = np.zeros(10)

    # target cost function
    def t(x, y, theta):
        return y - np.dot(x, theta)

    def dt(x, y, theta):
        return -x

    min_theta = minimize_stochastic(t, dt, x, y, theta)
    print('result:', sum(t(x, y, min_theta)))