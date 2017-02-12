from toolz import curry

dataset = [(1,1), (2,2), (3,3), (4,4)]
print(dataset)

Y_hat = lambda x, A, B: A*x +B
Y_hat = curry(Y_hat)

print(Y_hat(1,1,1))

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

def gradient_descent(learning_rate, A_point, df):
    return A_point - (df(A_point) * learning_rate)


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

old = Point(0,0)
new = Point(1,1)
convergent = 0.1

while abs(new.x-old.x) > convergent or abs(new.y - old.y) > convergent:
    new = gradient_descent(0.001, old, gradient_of_least_squers(datapoints))
    print (new, '->', old)
    print (abs(new.x-old.x))
    print (abs(new.x-old.x) > convergent)
    old = new

print (old)
print(list(map(str, map(lambda p: (Y_hat(p.x, old.x, old.y), p.y), datapoints))))
