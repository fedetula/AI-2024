import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import random


def simple_function(x):
    """
    Simple function to plot
    """
    return 3 * x**2 + 2 * x - 1


def dev_fun_alg(x):
    return 6 * x + 2


def dev_fun_num(x):
    delta = 0.01
    y2 = simple_function(x)
    y1 = simple_function(x + delta)

    return -(y2 - y1) / delta


def grad_desc(lr=0.01, steps=100):
    x = random.random()
    for _ in range(steps):
        grad_x = dev_fun_num(x)
        x = x - lr * grad_x
    return x


def compute_derivative(func, h=1e-7):
    def derivative(x):
        return (func(x + h) - func(x)) / h

    return derivative


def grad_desc_func(func, lr=0.01, steps=100):
    x = random.random()
    derivative = compute_derivative(func)
    for _ in range(steps):
        grad_x = derivative(x)
        x = x - lr * grad_x
    return x


x = np.linspace(-2, 2, 100)
y = simple_function(x)

n_updates = 200

x_opt = 2
y_opt = simple_function(x_opt)

plt.plot(x, y)
for i in range(n_updates):
    x_opt = x_opt - compute_derivative(simple_function)(x_opt) * 0.002
    y_opt = simple_function(x_opt)
    plt.plot(x_opt, y_opt, "ro")  # plot the current optimized point

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent Optimization")
plt.show()
