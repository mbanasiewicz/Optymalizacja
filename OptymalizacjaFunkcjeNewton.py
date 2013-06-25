from pprint import pprint
from matplotlib import pyplot as plt
from numpy import cos as ncos
import mpl_toolkits.mplot3d.axes3d as p3
import operator
from math import *
from matplotlib import cm
import numpy as np

class Beale:
    def narysujFunkcje(self, start, stop):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.2)
        y = np.arange(-4.5, 4.5, 0.2)
        [X, Y] = np.meshgrid(x,y)
        Z = (1.5 - X * ( 1 - Y) ) ** 2 + (2.25 - X * ( 1 - Y ** 2 ) ) ** 2 + ( 2.625 - X * ( 1 - Y ** 3 ) )**2
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([start[0]], [start[1]], [self.funkcja(start[0] , start[1])], color='g', s=20)
        ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop[0], stop[1])], color='r')
        ax.scatter([3], [0.5], [self.funkcja(3, 0.5)], color='b')
        plt.show()

    def funkcja(self, X, Y):
        return (1.5 - X * ( 1 - Y) ) ** 2 + (2.25 - X * ( 1 - Y ** 2 ) ) ** 2 + ( 2.625 - X * ( 1 - Y ** 3 ) )**2

    def gradient(self, x):

        return np.array([2 * (x[0] * (x[1]**6 + x[1]**4 - 2 * x[1]**3 - x[1]**2 - 2 * x[1] + 3) + 2.625 * x[1]**3 + 2.25 * x[1] ** 2 + 1.5 * x[1] - 6.375),
                        x[0] * ( x[0] * (6 * x[1]**5 + 4 * x[1] ** 3 - 6 * x[1]**2 - 2 * x[1] - 2) + 15.75 + x[1]**2 + 9 * x[1] + 3)])


    def hessian(self, x):
        a = np.zeros((2,2))
        a[0,0] = 2 * (x[1] ** 6 + x[1] ** 4 - 2 * x[1] ** 3 - x[1] ** 2 - 2 * x[1] + 3)
        a[0,1] = x[0] * (12 * x[1] ** 5+8 * x[1] ** 3 - 12 * x[1] ** 2 - 4 * x[1] - 4) + 15.75 * x[1] ** 2+9 * x[1] + 3
        a[1,0] = x[0] * (12 * x[1] ** 5+8 * x[1] ** 3 - 12 * x[1] ** 2 - 4 * x[1] - 4) + 15.75 * x[1] ** 2+9 * x[1] + 3
        a[1,1] = x[0] * (x[0] * (30 * x[1] ** 4 + 12 * x[1] ** 2 - 12 * x[1] - 2) + 31.5 * x[1] + 9)
        return a

class Rosenbrock:
    """
    Rosenbrock
    """
    def narysujFunkcje(self, start, stop):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-4.5, 4.5, 0.2)
        y = np.arange(-4.5, 4.5, 0.2)
        [X, Y] = np.meshgrid(x,y)
        Z = self.funkcja(X, Y)
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([start[0]], [start[1]], [self.funkcja(start[0] , start[1])], color='g', s=20)
        ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop[0], stop[1])], color='r')
        ax.scatter([1], [1], [self.funkcja(1, 1)], color='b')
        plt.show()

    def funkcja(self, X, Y):
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

class Hump:
    def narysujFunkcje(self, start, stop):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = np.arange(-5, 5, 0.2)
        y = np.arange(-5, 5, 0.2)

        [X, Y] = np.meshgrid(x,y)
        Z = self.funkcja(X, Y)
        ax.plot_wireframe(X,Y,Z)
        ax.scatter([start[0]], [start[1]], [self.funkcja(start[0] , start[1])], color='g', s=20)
        ax.scatter([stop[0]], [stop[1]], [self.funkcja(stop[0], stop[1])], color='r')
        ax.scatter([0.0898], [-0.7126], [self.funkcja(0.0898, -0.7126)], color='b')
        ax.scatter([-0.0898], [0.7126], [self.funkcja(-0.0898, 0.7126)], color='b')
        plt.show()

    def funkcja(self, X, Y):
        return 1.0316285 + 4 * X ** 2 - 2.1 * X ** 4 + X**6 / 3 + X*Y - 4 * Y ** 2 + 4 * Y ** 4

    def gradient(self, x):
        return np.array([2 * x[0] ** 5 - 8.4 * x[0] ** 3 + 8 * x[0] + x[1], x[0] + 16 * x[1]**3 - 8 * x[1]])

    def hessian(self, x):
        a = np.zeros((2,2))
        a[0,0] = 10 * x[0] ** 4 - 25.2 * x[0] ** 2 + 8
        a[0,1] = 1
        a[1,0] = 1
        a[1,1] =  48 * x[1]**2 - 8
        return a