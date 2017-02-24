from toolz import curry
import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt
from random import shuffle


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, value):
        return Point(value * self.x, value * self.y)

    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

    def __getitem__(self, key):
        if key == 0:
            return self.x

        return self.y

    def __str__(self):
        return 'Point(%f,%f)' % (self.x, self.y)


def gradient_descent_step(learning_rate, A_point, df):
    return A_point - (df(A_point) * learning_rate)


def least_squere_error(point, datapoints):
    return sum([(point.y + (point.x * p.x) - p.y)**2 for p in datapoints])/len(datapoints)


def gradient_descent(point, learning_rate, error_treshold, max_iterations, datapoints):
    E = 0
    for i in range(max_iterations):
        point = gradient_descent_step(
            learning_rate, point, gradient_of_least_squers(datapoints))

        e = least_squere_error(point, datapoints)
        if abs(E - e) > error_treshold:
            E = e
        else:
            return point

    return point


@curry
def gradient_of_least_squers(datapoints, point):
    A = point.x
    B = point.y
    return Point(
        x=sum(map(lambda p: (((A * p.x + B) - p.y) * p.x),
                  datapoints)) / len(datapoints),
        y=sum(map(lambda p: (((A * p.x + B) - p.y)), datapoints)) / len(datapoints)
    )


@curry
def stochastic_df(p, point):
    A = point.x
    B = point.y
    return Point(
        x=(((A * p.x + B) - p.y) * p.x),
        y=(((A * p.x + B) - p.y))
    )


def stochastic_gradient_descent(point, learning_rate, error_treshold, max_iterations, datapoints):
    E = 0
    for i in range(max_iterations):
        shuffle(datapoints)
        for p in datapoints:
            old = point
            point = gradient_descent_step(
                learning_rate, point, stochastic_df(p))

        e = least_squere_error(point, datapoints)
        if abs(E - e) > error_treshold:
            E = e
        else:
            return point

    return point


dataset = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
dataset = [
    (3.385,   44.500),
    (0.480,   15.500),
    (1.350,    8.100),
    # (  465.000,  423.000),
    # (   36.330,  119.500),
    # (   27.660,  115.000),
    # (   14.830,   98.200),
    (1.040,    5.500),
    (4.190,   58.000),
    (0.425,    6.400),
    (0.101,    4.000),
    (0.920,    5.700),
    (1.000,    6.600),
    (0.005,    0.140),
    (0.060,    1.000),
    (3.500,   10.800),
    (2.000,   12.300),
    (1.700,    6.300),
    # ( 2547.000, 4603.000),
    (0.023,    0.300),
    # (  187.100,  419.000),
    # (  521.000,  655.000),
    (0.785,    3.500),
    # (   10.000,  115.000),
    (3.300,   25.600),
    (0.200,    5.000),
    (1.410,   17.500),
    # (  529.000,  680.000),
    #  (  207.000,  406.000),
    (0.750,   12.300),
    # (   62.000, 1320.000),
    # ( 6654.000, 5712.000),
    (3.500,    3.900),
    # (    6.800,  179.000),
    # (   35.000,   56.000),
    (4.050,   17.000),
    (0.120,    1.000),
    (0.023,    0.400),
    (0.010,    0.250),
    # (    1.400,   12.500),
    # (  250.000,  490.000),
    # (    2.500,   12.100),
    # (   55.500,  175.000),
    # (  100.000,  157.000),
    # (   52.160,  440.000),
    # (   10.550,  179.500),
    (0.550,    2.400),
    # (   60.000,   81.000),
    (3.600,   21.000),
    (4.288,   39.200),
    (0.280,    1.900),
    (0.075,    1.200),
    (0.122,    3.000),
    (0.048,    0.330),
    # (  192.000,  180.000),
    (3.000,   25.000),
    # (  160.000,  169.000),
    (0.900,    2.600),
    (1.620,   11.400),
    (0.104,    2.500),
    (4.235,   50.400),

]

Y_hat = lambda x, A, B: A * x + B
Y_hat = curry(Y_hat)

datapoints = [Point(x[0], x[1]) for x in dataset]

B1B2 = gradient_descent(point=Point(0, 0),
                        learning_rate=0.0001,
                        error_treshold=0.001,
                        max_iterations=1000,
                        datapoints=datapoints)
print(B1B2)

B1B2_b = stochastic_gradient_descent(point=Point(0, 0),
                                     learning_rate=0.001,
                                     error_treshold=0.001,
                                     max_iterations=1000,
                                     datapoints=datapoints)
print(B1B2_b)

Y_hat_datapoints = list(
    map(lambda p: (p.x, Y_hat(p.x, B1B2.x, B1B2.y)), datapoints))
Y_hat_datapoints_b = list(
    map(lambda p: (p.x, Y_hat(p.x, B1B2_b.x, B1B2_b.y)), datapoints))

"PLOT:"


def to_x_y(dataset):
    return [[t[0] for t in dataset], [t[1] for t in dataset]]

"Plot batch gradient descent"
plt.subplot(2, 2, 1)
plt.title("Gradient Descent")
plt.plot(*to_x_y(dataset), 'r.')
plt.plot(*to_x_y(Y_hat_datapoints), 'g-')

"Plot stochastic gradient descent"
plt.subplot(2, 2, 2)
plt.title("Stochastic Gradient Descent")
plt.plot(*to_x_y(dataset), 'r.')
plt.plot(*to_x_y(Y_hat_datapoints_b), 'y--')
plt.axis(xmin=0, ymin=0)

sin_x = np.linspace(0, pi*2, 25)
sin_y = [sin(x) for x in sin_x]
sin_y += np.random.normal(scale=0.1, size=25)

"Plot sinusoid data"
plt.subplot(2, 2, 3)
plt.title("Polynomial")
plt.plot(sin_x, sin_y, 'r.')
plt.savefig('line.png')
