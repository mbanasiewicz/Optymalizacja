from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import operator
import math
from matplotlib import cm
from OptymalizacjaFunkcjeHookJeeves import *
class HookJeeves:
    """
    Klasa optymalizuje wybrana funkcje metoda bezgradientowa Hookea Jeevsa
    """
    debugMode = False
    # eps
    eps = 0.0001
    x_start = [0, 0]
    # Punkt poczatkowy
    x_zero = [0, 0]
    # Punkt dla kroku roboczego
    x_k = [0,0]
    # Punkt bazowy
    x_b = [0,0]
    x_b_zero = [0,0]
    # Wektory ortogonalne
    wektory = [[1,0], [0,1]]
    # n - czyli ilosc zmiennych - wektorow ortog
    n = 2
    # k - indeks obecnego wektora
    k = 0
    # wartosc w kroku roboczym
    q_zero = 0.0
    q = 0.0
    # ilosc iteracji
    ii = 0
    # wspolczynnik zmniejszajacy
    wsp_zmn = 0.0
    obiektFunkcji = None
    x_tmps = []
    y_tmps = []
    z_tmps = []
    show_plot = False
    tmpStartowy = []
    pokaz_sciezke = False
    def __init__(self, x_zero, wsp_zmn, eps, pokaz_sciezke, debug_mode, show_plot, obiekt_funkcji):
        # na starcie obliczamy wartosc w pkt poczatkowym
        self.tmpStartowy = x_zero
        self.x_zero = x_zero
        self.x_start = x_zero
        self.wsp_zmn = wsp_zmn
        self.x_b = x_zero
        self.x_b_zero = x_zero
        self.eps = eps
        self.pokaz_sciezke = pokaz_sciezke
        self.debugMode = debug_mode
        self.show_plot = show_plot
        self.obiektFunkcji = obiekt_funkcji

    def probny(self):
        self.k = 0
        self.q_zero = self.funkcja(self.x_zero)
        # wartosc w pkt startowym
        if self.debugMode:
            print '\n#########START -> ' + str(self.q_zero) + ' Iteracja: ' + str(self.ii) + ' X_ZERO ->' + str(self.x_zero)
        while self.k < self.n: # jezeli nie sprawdzilismy we wszystkich kierunkach
            old_x_k = self.x_k
            self.x_k = self.punktOWektor(self.x_zero, self.wektory[self.k], False)
            self.q = self.funkcja(self.x_k)
            if self.debugMode:
                print 'Q -> ' + str(self.q) + ' -> ' + str(self.x_k) + ' k -> ' + str(self.k)

            if self.q < self.q_zero: # jezeli w tym kroku probnym wartosc funkcji jest mniejsza to q_zero = q
                self.q_zero = self.q
            else:
                # w przeciwnym wypadku liczymy w kierunku przeciwnym
                self.x_k = self.punktOWektor(self.x_zero, self.wektory[self.k], True)
                if self.funkcja(self.x_k) < self.q_zero:
                    self.q = self.funkcja(self.x_k)
                    if self.debugMode:
                        print 'Q -> ' + str(self.q) + ' -> ' + str(self.x_k) + ' inv k -> ' + str(self.k)
                else:
                    if self.debugMode:
                        print 'Q -> ' + str(self.q) + ' -> ' + str(self.x_k) + ' inv k -> ' + str(self.k) + 'BAD'
                    self.x_k = old_x_k
            self.k += 1

        # print 'END SEARCH X_K -> ' + str(self.x_k) + ' Q -> ' + str(self.q)
        if self.debugMode:
            print 'Got this ->' + str(self.x_k)
        # krok 7
        if self.funkcja(self.x_b_zero) > self.funkcja(self.x_k):
            self.x_b = self.x_k
            # dalej etap roboczy
        else:
            if self.ii == 1:
                print 'Zmien punkt startowy'
            else:
                self.x_zero = self.x_start
                self.wsp_zmn *= 0.99
                print 'Zmniejsz -> ' + str(self.wsp_zmn)

    def funkcja(self, punkt):
        return self.obiektFunkcji.funkcja(punkt)
    def narysujFunkcje(self):
        return self.obiektFunkcji.narysujFunkcje(self.x_tmps, self.y_tmps, self.z_tmps, self.x_b_zero, self.tmpStartowy, self.pokaz_sciezke)
    def roboczy(self):
        self.ii += 1
        self.x_zero = self.roznicaPunktow(self.punktRazyDwa(self.x_b),self.x_b_zero)
        self.x_b_zero = self.x_b
    def optymalizuj(self):
        while self.wsp_zmn > self.eps and self.ii < 10000:
            self.probny()
            self.roboczy()
            self.x_tmps.append(self.x_zero[0])
            self.y_tmps.append(self.x_zero[1])
            self.z_tmps.append(self.funkcja(self.x_b_zero))
        print (self.ii, self.x_b_zero, self.funkcja(self.x_b_zero))
        if self.show_plot:
            self.narysujFunkcje()
        return (self.ii, self.x_b_zero, self.funkcja(self.x_b_zero))

    # Zwraca punkt przesuniety o wektor
    def punktOWektor(self, punkt, wektor, przeciwny):
        if not przeciwny:
            return [punkt[0] + self.wsp_zmn * wektor[0], punkt[1] + self.wsp_zmn * wektor[1]]
        else:
            return [punkt[0] - 2 * self.wsp_zmn * wektor[0], punkt[1] - 2 * self.wsp_zmn * wektor[1]]
    #funkcja dla kroku roboczego
    def punktRazyDwa(self, punkt):
        return [x * 2 for x in punkt]
    # funkcja dla kroku roboczego
    def roznicaPunktow(self, a, b):
        return [operator.sub(a,b) for a,b in zip(a, b)]



if __name__ == "__main__":
    fh = Hump()
    fb = Beale()
    fe = Easom()
    fg = Goldstein() # -1, 2
    fs = Sphere()
    fr = Rosenbrock()
    jeev = HookJeeves(x_zero=[-1, 2] ,eps=0.0001, wsp_zmn=0.4, debug_mode=False, show_plot=True, pokaz_sciezke=False,obiekt_funkcji=fg)
    print jeev.optymalizuj()