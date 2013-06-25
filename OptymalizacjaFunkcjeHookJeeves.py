from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import operator
import math
from matplotlib import cm

class Sphere:
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero,tmpStartowy,pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-10, 10, 0.5)
        y = np.arange(-10, 10, 0.5)
        [X, Y] = np.meshgrid(x,y)
        Z = X ** 2 + Y ** 2
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero)], color='g', s=20)
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy)], color='r')
        plt.show()

    def funkcja(self, point):
        return point[0] ** 2 + point[1] ** 2


class Goldstein:
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero,tmpStartowy,pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-2, 2, 0.2)
        y = np.arange(-2, 2, 0.2)
        [X, Y] = np.meshgrid(x,y)
        A = 1+( ( X + Y + 1 ) ** 2 ) * ( 19 - 14 * X + 3 * Y ** 2 - 14 * Y + 6 * X * Y + 3 * Y ** 2);
        B = 30 + (( 2 * X - 3 * Y) ** 2) * ( 18 - 32 * X + 12 * X ** 2 + 48 * Y - 36 * X * Y + 27 * Y ** 2);
        Z = A * B;
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero)], color='g', s=20)
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy)], color='r')
        plt.show()

    def funkcja(self, point, Y=None):
        if Y:
            X = point
            point = [X, Y]
        A = 1 +( ( point[0] + point[1] + 1 ) ** 2 ) * ( 19 - 14 * point[0] + 3 * point[1] ** 2 - 14 * point[1] + 6 * point[0] * point[1] + 3 * point[1] ** 2)
        B = 30 + (( 2 * point[0] - 3 * point[1]) ** 2) * ( 18 - 32 * point[0] + 12 * point[0] ** 2 + 48 * point[1] - 36 * point[0] * point[1] + 27 * point[1] ** 2);
        return A * B

class Easom():
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero,tmpStartowy,pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-100, 100, 2)
        y = np.arange(-100, 100, 2)
        [X, Y] = np.meshgrid(x,y)
        Z = -1 * np.cos(X)* np.cos(Y) * np.exp(-1 * ( X - math.pi) ** 2 - (Y - math.pi) ** 2);
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero)], color='g', s=20)
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy)], color='r')
        plt.show()

    def funkcja(self, point):
        ret = -1 * np.cos(point[0]) * np.cos(point[1]) * np.exp(-1 * ( point[0] - math.pi) ** 2 - (point[1]-math.pi) ** 2)
        return ret

class Beale:
    """
    (1.5 - x + (x * y)) ** 2 + (2.25 - x + (x * y ** 2)) ** 2 + (2.625 - x + (x * y ** 3)) ** 2 beale
    """
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero,tmpStartowy,pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.2)
        y = np.arange(-4.5, 4.5, 0.2)
        [X, Y] = np.meshgrid(x,y)
        Z = (1.5 - X * ( 1 - Y) ) ** 2 + (2.25 - X * ( 1 - Y ** 2 ) ) ** 2 + ( 2.625 - X * ( 1 - Y ** 3 ) )**2
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy[0] , tmpStartowy[1])], color='g', s=20)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero[0], x_b_zero[1])], color='r')
        ax.scatter([3], [0.5], [self.funkcja(3, 0.5)], color='b')
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

class Hump:
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero, tmpStartowy, pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-5, 5, 0.2)
        y = np.arange(-5, 5, 0.2)

        [X, Y] = np.meshgrid(x,y)
        Z = self.funkcja(X, Y)
        ax.plot_wireframe(X,Y,Z)
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy[0] , tmpStartowy[1])], color='g', s=20)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero[0], x_b_zero[1])], color='r')
        ax.scatter([0.0898], [-0.7126], [self.funkcja(0.0898, -0.7126)], color='b')
        ax.scatter([-0.0898], [0.7126], [self.funkcja(-0.0898, 0.7126)], color='b')
        plt.show()

    def funkcja(self, X, Y=None):
        if Y == None:
            Y = X[1]
            X = X[0]
        return 1.0316285 + 4 * X ** 2 - 2.1 * X ** 4 + X**6 / 3 + X*Y - 4 * Y ** 2 + 4 * Y ** 4;

    def gradient(self, x):
        return np.array([2 * x[0] ** 5 - 8.4 * x[0] ** 3 + 8 * x[0] + x[1], x[0] + 16 * x[1]**3 - 8 * x[1]])

    def hessian(self, x):
        a = np.zeros((2,2))
        a[0,0] = 10 * x[0] ** 4 - 25.2 * x[0] ** 2 + 8
        a[0,1] = 1
        a[1,0] = 1
        a[1,1] =  48 * x[1]**2 - 8
        return a

class Rosenbrock:
    def narysujFunkcje(self, x_tmps, y_tmps, z_tmps, x_b_zero, tmpStartowy, pokaz_sciezke):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.2)
        y = np.arange(-4.5, 4.5, 0.2)
        [X, Y] = np.meshgrid(x,y)
        Z = self.funkcja(X, Y)
        ax.plot_wireframe(X,Y,Z)
        if pokaz_sciezke:
            ax.scatter(x_tmps, y_tmps, z_tmps, color='y', alpha=1)
        ax.scatter([tmpStartowy[0]], [tmpStartowy[1]], [self.funkcja(tmpStartowy[0] , tmpStartowy[1])], color='g', s=20)
        ax.scatter([x_b_zero[0]], [x_b_zero[1]], [self.funkcja(x_b_zero[0], x_b_zero[1])], color='r')
        ax.scatter([1], [1], [self.funkcja(1, 1)], color='b')
        plt.show()

    def funkcja(self, X, Y=None):
        if Y == None:
            Y = X[1]
            X = X[0]
        return (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    def gradient(self, x):
        return np.array([2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)])

    def hessian(self, x):
        a = np.zeros((2,2))
        a[0,0] = (1200 * (x[0] ** 2) - 400 * x[1] + 2)
        a[0,1] = - 400 * x[0]
        a[1,0] = - 400 * x[0]
        a[1,1] = 200
        return a