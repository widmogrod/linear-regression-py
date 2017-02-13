from toolz import curry

dataset = [(1,1), (2,2), (3,3), (4,4)]
print(dataset)

Y_hat = lambda x, A, B: A*x +B
Y_hat = curry(Y_hat)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, value):
        return Point(value * self.x, value * self.y)

    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

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

datapoints = list(map(apply_tuple(Point), dataset))
print(list(map(str, datapoints)))

convergent = 0.0000001
new = gradient_descent(Point(0,0), convergent)

print (new)
print(list(map(str, map(lambda p: (Y_hat(p.x, new.x, new.y), p.y), datapoints))))
