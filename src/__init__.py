from toolz import curry
import matplotlib.pyplot as plt

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

def gradient_descent(point, convergent):
    old = point
    point = gradient_descent_step(0.01, point, gradient_of_least_squers(datapoints))

    is_convergent = lambda point, old: abs(point.x-old.x) < convergent and abs(point.y - old.y) < convergent

    while not is_convergent(point, old):
        old = point
        point = gradient_descent_step(0.01, point, gradient_of_least_squers(datapoints))

    return point

@curry
def gradient_of_least_squers(datapoints, point):
    A = point.x
    B = point.y
    return Point(
            x=2*sum(map(lambda p: (((A * p.x + B) - p.y) * p.x), datapoints)),
            y=2*sum(map(lambda p: (((A * p.x + B) - p.y)), datapoints))
    )

apply_tuple = lambda f, t: f(*t)
apply_tuple = curry(apply_tuple)

dataset = [(1,2), (2,3), (3,4), (4,5), (5,6)]

Y_hat = lambda x, A, B: A*x +B
Y_hat = curry(Y_hat)

datapoints = list(map(apply_tuple(Point), dataset))
print(list(map(str, datapoints)))

convergent = 0.0000001
B1B2 = gradient_descent(Point(0,0), convergent)

print(B1B2)
print(list(map(str, map(lambda p: (Y_hat(p.x, B1B2.x, B1B2.y), p.y), datapoints))))

Y_hat_datapoints=list(map(lambda p: (p.x, Y_hat(p.x, B1B2.x, B1B2.y)), datapoints))

"PLOT:"
def to_x_y(dataset):
    return [[t[0] for t in dataset]
           ,[t[1] for t in dataset]]

plt.plot(*to_x_y(dataset), 'r.')
plt.plot(*to_x_y(Y_hat_datapoints), 'g--')
plt.axis(xmin=0, ymin=0)
plt.savefig('line.png')
