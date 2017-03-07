from toolz import curry
from math import sqrt
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

    def l2(self):
        return sqrt((self.x * self.x) + (self.y * self.y))


class Point4:

    def __init__(self, x, y, z, q):
        self.x = x
        self.y = y
        self.z = z
        self.q = q

    def __mul__(self, value):
        return Point4(value * self.x, value * self.y, value * self.z, value * self.q)

    def __sub__(self, point):
        return Point4(self.x - point.x, self.y - point.y, self.z - point.z, self.q - point.q)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        if key == 2:
            return self.z

        return self.q

    def __str__(self):
        return 'Point4(%f,%f,%f,%f)' % (self.x, self.y, self.z, self.q)

    def l2(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2 + self.q**2)


class Result:
    def __init__(self, point, error):
        self.point = point
        self.error = error

    def __str__(self):
        return 'R(%s,%f)' % (self.point, self.error)


@curry
def gradient_descent_step(learning_rate, point, df):
    return point - (df(point=point) * learning_rate)


@curry
def least_squere_error_method(Y_hat, coefficients, datapoints):
    return sum([(Y_hat(p, coefficients) - p.y)**2 for p in datapoints]) / len(datapoints)

def least_squere_error(point, datapoints):
    def Y_hat(p, coefficients):
        return coefficients[0]*p.x**0 + coefficients[1]*p.x**1

    return least_squere_error_method(Y_hat, point, datapoints)


@curry
def least_squere_error_polynomial(point, datapoints):
    def Y_hat(p, coefficients):
        return coefficients[0]*p.x**0 \
             + coefficients[1]*p.x**1 \
             + coefficients[2]*p.x**2 \
             + coefficients[3]*p.x**3

    return least_squere_error_method(Y_hat, point, datapoints)


@curry
def least_squere_error_polynomial_with_l2(point, datapoints, bias_coefficient=1):
    bias = bias_coefficient * point.l2()
    return least_squere_error_polynomial(point, datapoints) + bias


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
        x=sum(map(lambda p: (((A * p.x + B) - p.y) * p.x), datapoints)) / len(datapoints),
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
            point = gradient_descent_step(
                learning_rate, point, stochastic_df(p))

        e = least_squere_error(point, datapoints)
        if abs(E - e) > error_treshold:
            E = e
        else:
            return point

    return point



@curry
def gradient_of_least_squers_polynomial(datapoints, point):
    A = point.x
    B = point.y
    C = point.z
    D = point.q
    return Point4(
        x=sum(map(lambda p: (((A + (p.x*B) + (p.x**2*C) + (p.x**3*D)) - p.y) * p.x**0), datapoints)) / len(datapoints),
        y=sum(map(lambda p: (((A + (p.x*B) + (p.x**2*C) + (p.x**3*D)) - p.y) * p.x**1), datapoints)) / len(datapoints),
        z=sum(map(lambda p: (((A + (p.x*B) + (p.x**2*C) + (p.x**3*D)) - p.y) * p.x**2), datapoints)) / len(datapoints),
        q=sum(map(lambda p: (((A + (p.x*B) + (p.x**2*C) + (p.x**3*D)) - p.y) * p.x**3), datapoints)) / len(datapoints)
    )


@curry
def gradient_of_least_squers_polynomial_with_l2(datapoints, point, bias_coefficient=1):
    l2 = point.l2()
    p = gradient_of_least_squers_polynomial(datapoints, point)
    return Point4(
            x=p[0] + point[0]/l2 * bias_coefficient,
            y=p[1] + point[1]/l2 * bias_coefficient,
            z=p[2] + point[2]/l2 * bias_coefficient,
            q=p[3] + point[3]/l2 * bias_coefficient
            )


def generic_gradient_descent(gradient_func, error_func, point,
                             learning_rate, error_treshold, max_iterations, datapoints):
    E = 0
    for i in range(max_iterations):
        point = gradient_descent_step(
            learning_rate, point, gradient_func(datapoints))

        e = error_func(point, datapoints)
        if abs(E - e) > error_treshold:
            E = e
        else:
            return point

    return point


@curry
def gd(gradient_func, error_func, point):
    while(True):
        point = gradient_func(point=point)
        error = error_func(point=point)
        yield Result(point, error)


@curry
def max_iterations(iterator, max=1000):
    for index, value in enumerate(iterator):
        if index < max:
            yield value
        else:
            break


@curry
def learning(iterator, error_treshold=0.00001, E=0):
    for result in iterator:
        if abs(E-result.error) > error_treshold:
            E = result.error
            yield result
        else:
            break


x_values = range(6)
y_values = x_values + np.random.normal(scale=0.8, size=len(x_values))
dataset = list(zip(x_values, y_values))


Y_hat = lambda x, A, B: A * x + B
Y_hat = curry(Y_hat)


datapoints = [Point(x[0], x[1]) for x in dataset]

B1B2 = gradient_descent(point=Point(0, 0),
                        learning_rate=0.0001,
                        error_treshold=0.001,
                        max_iterations=1000,
                        datapoints=datapoints)
print("B1B2 gradient descent=", B1B2)

B1B2_b = stochastic_gradient_descent(point=Point(0, 0),
                                     learning_rate=0.001,
                                     error_treshold=0.001,
                                     max_iterations=1000,
                                     datapoints=datapoints)

print("B1B2 stochstic gradient descent=", B1B2_b)

Y_hat_datapoints = list(
    map(lambda p: (p.x, Y_hat(p.x, B1B2.x, B1B2.y)), datapoints))
Y_hat_datapoints_b = list(
    map(lambda p: (p.x, Y_hat(p.x, B1B2_b.x, B1B2_b.y)), datapoints))


"Poly - start"
Y_hat_poly = lambda x, c: (x**0*c[0] + x**1*c[1] + x**2*c[2] + x**3*c[3])
Y_hat_poly = curry(Y_hat_poly)

samples=3

sin_x_independent = np.linspace(-pi, pi, samples)
sin_y_dependent = [sin(x) for x in sin_x_independent]
sin_y_dependent += np.random.normal(scale=0.8, size=samples)

sin_x_plot = np.arange(-pi*2, pi*2, 0.1)
poly_dataset = list(zip(sin_x_independent, sin_y_dependent))
poly_datapoints = [Point(*x) for x in poly_dataset]

# P4 = generic_gradient_descent(point=Point4(1,1,1,1),
#                               learning_rate=0.001,
#                               error_treshold=0.000001,
#                               max_iterations=10000,
#                               datapoints=poly_datapoints,
#                               gradient_func=gradient_of_least_squers_polynomial,
#                               error_func=least_squere_error_polynomial)
#

pipeline = max_iterations(learning(gd(
    point=Point4(1, 1, 1, 1),
    gradient_func=gradient_descent_step(
        learning_rate=0.001,
        df=gradient_of_least_squers_polynomial(
            datapoints=poly_datapoints
        )
    ),
    error_func=least_squere_error_polynomial(
        datapoints=poly_datapoints
    )
), error_treshold=0.01), max=100)

pipeline = list(pipeline)

P4 = pipeline[-1].point

print("P4 gradient descent polynomial =", P4)
Y_hat_poly_datapoints_1 = list(map(lambda x: (x, Y_hat_poly(x, P4)), sin_x_plot))

P5 = generic_gradient_descent(point=Point4(1, 1, 1, 1),
                              learning_rate=0.001,
                              error_treshold=0.000001,
                              max_iterations=10000,
                              datapoints=poly_datapoints,
                              gradient_func=gradient_of_least_squers_polynomial_with_l2(
                                  bias_coefficient=0.2),
                              error_func=least_squere_error_polynomial_with_l2(
                                  bias_coefficient=0.2))

print("P5 gradient descent polynomial L2 =", P5)
Y_hat_poly_datapoints_2 = list(map(lambda x: (x, Y_hat_poly(x, P5)), sin_x_plot))
"Poly = end"


"PLOT:"


def to_x_y(dataset):
    return list(zip(*dataset))


"Plot batch gradient descent"
plt.subplot(3, 2, 1)
plt.title("Gradient Descent")
plt.plot(*to_x_y(dataset), 'r.')
plt.plot(*to_x_y(Y_hat_datapoints), 'y')

"Plot stochastic gradient descent"
plt.subplot(3, 2, 2)
plt.title("Stochastic Gradient Descent")
plt.plot(*to_x_y(dataset), 'r.')
plt.plot(*to_x_y(Y_hat_datapoints_b), 'y')
plt.axis(xmin=0, ymin=0)

"Plot sinusoid data polynomial"
plt.subplot(3, 2, 3)
plt.title("Polynomial")
plt.ylim((-2, 2))
plt.plot(*to_x_y(poly_dataset), 'r.')
plt.plot(*to_x_y(Y_hat_poly_datapoints_1), 'y')

plt.subplot(3, 2, 5)
plt.title("Polynomial Error")
plt.ylim((0, 50))
plt.plot([r.error for r in pipeline], 'y')

"Plot sinusoid data polynomial L2"
plt.subplot(3, 2, 4)
plt.title("Polynomial L2")
plt.ylim((-2, 2))
plt.plot(*to_x_y(poly_dataset), 'r.')
plt.plot(*to_x_y(Y_hat_poly_datapoints_2), 'y')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('line.png')
