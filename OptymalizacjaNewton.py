import numpy as np
from OptymalizacjaFunkcjeNewton import *

class Newton:
    obiektFunkcji = None
    xZero = np.array([0,0])
    k = 0
    eps = 0.0001
    narysuj_funkcje = False
    def __init__(self, obiekt_funkcji, x_zero, eps, narysuj_funkcje):
        self.obiektFunkcji = obiekt_funkcji
        self.xZero = x_zero
        self.eps = eps
        self.narysuj_funkcje = narysuj_funkcje

    def optymalizuj(self):
        # Startup
        g = -1 * np.dot(np.linalg.inv(self.obiektFunkcji.hessian(self.xZero)),self.obiektFunkcji.gradient((self.xZero)))
        xnew = self.xZero + g
        xold = [np.inf, np.inf]
        ii = 0
        while np.linalg.norm(g) > self.eps:
            # print np.linalg.norm(self.obiektFunkcji.gradient(xnew))
            if self.obiektFunkcji.funkcja(xnew[0], xnew[1]) < self.obiektFunkcji.funkcja(xold[0], xold[1]):
                print 'Mniejsza' + str(self.obiektFunkcji.funkcja(xnew[0], xnew[1]) - self.obiektFunkcji.funkcja(xold[0], xold[1]))
            ii += 1
            x = xnew
            g = -1 * np.dot(np.linalg.inv(self.obiektFunkcji.hessian(x)),self.obiektFunkcji.gradient(x))

            xnew = x + g
            xold = x
        print (ii, xnew, self.obiektFunkcji.funkcja(xnew[0], xnew[1]))
        if self.narysuj_funkcje:
            self.obiektFunkcji.narysujFunkcje(self.xZero, xnew)
	return (ii, xnew, self.obiektFunkcji.funkcja(xnew[0], xnew[1]))

if __name__ == '__main__':
    fb = Beale()
    fr = Rosenbrock()
    fh = Hump()
    ntn = Newton(obiekt_funkcji=fh, x_zero=np.array([-50, 50]), eps = 0.00000000000001, narysuj_funkcje=True)
    print str(ntn.optymalizuj())