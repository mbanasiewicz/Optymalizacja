import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

# FUNKCJE
class Sphere:
    """
    x ** 2 + y ** 2 - inf parabolid
    """
    def narysujFunkcje(self, start=None, stop=None):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-10, 10, 0.7)
        y = np.arange(-10, 10, 0.7)
        [X, Y] = np.meshgrid(x,y)
        Z = X ** 2 + Y ** 2
        ax.plot_wireframe(X,Y,Z)
        if start:
            ax.scatter([start[0]], [start[1]], [self.funkcja(start)], color='g', s=20)
        if stop:
            ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop)], color='r')
        # optimum
        ax.scatter([0], [0], [self.funkcja([0, 0])], color='b')
        plt.show()

    def funkcja(self, point):
        return point[0] ** 2 + point[1] ** 2
class Goldstein:
    def narysujFunkcje(self, start=None, stop=None):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-2, 2, 0.1)
        y = np.arange(-2, 2, 0.1)
        [X, Y] = np.meshgrid(x,y)
        A = 1+( ( X + Y + 1 ) ** 2 ) * ( 19 - 14 * X + 3 * Y ** 2 - 14 * Y + 6 * X * Y + 3 * Y ** 2);
        B = 30 + (( 2 * X - 3 * Y) ** 2) * ( 18 - 32 * X + 12 * X ** 2 + 48 * Y - 36 * X * Y + 27 * Y ** 2);
        Z = A * B;
        ax.plot_wireframe(X,Y,Z)
        if start:
            ax.scatter([start[0]], [start[1]], [self.funkcja(start)], color='g', s=20)
        if stop:
            ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop)], color='r')
        # optimum
        ax.scatter([0], [1], [self.funkcja([0, 1])], color='b')
        plt.show()

    def funkcja(self, point):
        A = 1 +( ( point[0] + point[1] + 1 ) ** 2 ) * ( 19 - 14 * point[0] + 3 * point[1] ** 2 - 14 * point[1] + 6 * point[0] * point[1] + 3 * point[1] ** 2)
        B = 30 + (( 2 * point[0] - 3 * point[1]) ** 2) * ( 18 - 32 * point[0] + 12 * point[0] ** 2 + 48 * point[1] - 36 * point[0] * point[1] + 27 * point[1] ** 2);
        return A * B
class Easom:
    """
    Easom
    """
    def narysujFunkcje(self, start=None, stop=None):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-100, 100, 2)
        y = np.arange(-100, 100, 2)
        [X, Y] = np.meshgrid(x,y)
        Z = -1 * np.cos(X)* np.cos(Y) * np.exp(-1 * ( X - math.pi) ** 2 - (Y - math.pi) ** 2)
        ax.plot_wireframe(X,Y,Z)
        if start:
            ax.scatter([start[0]], [start[1]], [self.funkcja(start)], color='g', s=20)
        if stop:
            ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop)], color='r')
        ax.scatter([np.pi], [np.pi], [self.funkcja([np.pi, np.pi])], color='b')
        plt.show()

    def funkcja(self, point):
        ret = -1 * np.cos(point[0]) * np.cos(point[1]) * np.exp(-1 * ( point[0] - math.pi) ** 2 - (point[1]-math.pi) ** 2)
        return ret
class Beale:
    def narysujFunkcje(self, start=None, stop=None):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.2)
        y = np.arange(-4.5, 4.5, 0.2)
        [X, Y] = np.meshgrid(x,y)
        Z = (1.5 - X * ( 1 - Y) ) ** 2 + (2.25 - X * ( 1 - Y ** 2 ) ) ** 2 + ( 2.625 - X * ( 1 - Y ** 3 ) )**2
        ax.plot_wireframe(X,Y,Z)
        if start:
            ax.scatter([start[0]], [start[1]], [self.funkcja(start)], color='g', s=20)
        if stop:
            ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop)], color='r')
        ax.scatter([3], [0.5], [self.funkcja([3, 0.5])], color='b')
        plt.show()

    def funkcja(self, X, Y=None):
        if Y == None:
            Y = X[1]
            X = X[0]
        return (1.5 - X * ( 1 - Y) ) ** 2 + (2.25 - X * ( 1 - Y ** 2 ) ) ** 2 + ( 2.625 - X * ( 1 - Y ** 3 ) )**2

    def gradient(self, x):
        return np.array([2 * (x[0] * (x[1] ** 6 + x[1] ** 4 - 2 * x[1] ** 3 - x[1] ** 2 - 2 * x[1] + 3) + 2.625 * x[1] ** 3 + 2.25 * x[1] ** 2 + 1.5 * x[1] - 6.375),
                      6 * x[0] * (x[0] * (x[1]**5 + 0.666667 * x[1] ** 3 - x[1] ** 2 - 0.333333 * x[1] - 0.333333) + 2.625 * x[1] ** 2 + 1.5 * x[1] + 0.5)])

    def hessian(self, x):
        a = np.zeros((2,2))
        a[0,0] = 2 * (x[1] ** 6 + x[1] ** 4 - 2 * x[1] ** 3 - x[1] ** 2 - 2 * x[1] + 3)
        a[0,1] = x[0] * (12 * x[1] ** 5+8 * x[1] ** 3 - 12 * x[1] ** 2 - 4 * x[1] - 4) + 15.75 * x[1] ** 2+9 * x[1] + 3
        a[1,0] = x[0] * (12 * x[1] ** 5+8 * x[1] ** 3 - 12 * x[1] ** 2 - 4 * x[1] - 4) + 15.75 * x[1] ** 2+9 * x[1] + 3
        a[1,1] = x[0] * (x[0] * (30 * x[1] ** 4 + 12 * x[1] ** 2 - 12 * x[1] - 2) + 31.5 * x[1] + 9)
        return a
class Rosenbrock:
    def narysujFunkcje(self, start=None, stop=None):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.9)
        y = np.arange(-4.5, 4.5, 0.9)
        [X, Y] = np.meshgrid(x,y)
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        ax.plot_wireframe(X,Y,Z)
        if start:
            ax.scatter([start[0]], [start[1]], [self.funkcja(start)], color='g', s=20)
        if stop:
            ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop)], color='r')
        # optimum dla 2d
        ax.scatter([1], [1], [self.funkcja([1, 1])], color='b')
        plt.show()

    def funkcja(self, X):
        return (1 - X[0]) ** 2 + 100 * (X[1] - X[0] ** 2) ** 2

    def gradient(self, x):
        return np.array([2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)])

    def hessian(self, x):
        a = np.zeros((2,2))
        print str((1200 * (x[0] ** 2) - 400 * x[1] + 2)) + '!'
        a[0,0] = (1200 * (x[0] ** 2) - 400 * x[1] + 2)
        a[0,1] = - 400 * x[0]
        a[1,0] = - 400 * x[0]
        a[1,1] = 200
        return a
