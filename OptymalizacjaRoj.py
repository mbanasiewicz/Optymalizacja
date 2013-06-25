import random
import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from OptymalizacjaFunkcjeRoj import *

class Swarm:
    iloscIteracji = 1000
    iloscOsobnikowRoju = 100
    # skrajne wartosci dla pozycji i polozecznia czasteczki
    predkoscMaksymalna = 20
    predkoscMinimalna = -20

    pozycjaMaksymalna = 20
    pozycjaMinimalna = -20
    obiektFunkcji = None
    rysujWykres = False

    def __init__(self, iloscIteracji=None, iloscOsobnikowRoju=None, 
    predkoscMaksymalna=None, predkoscMinimalna=None, 
    obiektFunkcji=None, pozycjaMaksymalna=None, 
    pozycjaMinimalna = None, rysujWykres=False):
        if iloscIteracji:
            self.iloscIteracji = iloscIteracji
        if iloscOsobnikowRoju:
            self.iloscOsobnikowRoju = iloscOsobnikowRoju
        if predkoscMaksymalna:
            self.predkoscMaksymalna = predkoscMaksymalna
        if predkoscMinimalna:
            self.predkoscMinimalna = predkoscMinimalna
        if obiektFunkcji:
            self.obiektFunkcji = obiektFunkcji
        if pozycjaMaksymalna:
            self.pozycjaMaksymalna = pozycjaMaksymalna
        if pozycjaMinimalna:
            self.pozycjaMinimalna = pozycjaMinimalna
        if rysujWykres:
            self.rysujWykres = rysujWykres

    def funkcja(self, x):
        if self.obiektFunkcji:
            return self.obiektFunkcji.funkcja(x)
        else:
            print 'WSTRZYKNIJ OBIEKT FUNKCJI'
            exit()

    # glowna funkcja
    def optymalizuj(self):
      # Stworzenie roju osobnikow
      roj = []
      for i in range(self.iloscOsobnikowRoju):
        roj.append(Osobnik(self.predkoscMaksymalna, self.predkoscMinimalna, self.pozycjaMaksymalna, self.pozycjaMinimalna, self))
      # inicjujemy najlepsza predkosc polozenie i blad
      najlepszaPozycjaRoju = []
      najlepszaPredkoscRoju = []
      najlepszaWartoscRoju = -1
      # petla po ilosci iteracji
      for i in range(self.iloscIteracji):
        for osobnik in roj: # iteracja po roju i ustalenie ich pozycji
          wartoscOsobnika = osobnik.wartoscOsobnika()
          # jezeli to jest najlepszy osobnik z roju to zapisujemy jej pozycje i predkosc
          if wartoscOsobnika < najlepszaWartoscRoju or najlepszaWartoscRoju == -1:
            najlepszaPozycjaRoju = [osobnik.listaPozycji[-1][0], osobnik.listaPozycji[-1][1]]
            najlepszaPredkoscRoju = [osobnik.listaPredkosci[-1][0], osobnik.listaPredkosci[-1][1]]
            najlepszaWartoscRoju = wartoscOsobnika

        # uaktualnienie wszystkich elementow roju wzgledem najnowszej najlepszej pozycji
        for osobnik in roj:
          osobnik.zmienPredkosc(najlepszaPozycjaRoju)
          osobnik.zmienPozycje()
      print (i, najlepszaWartoscRoju, najlepszaPozycjaRoju)
      if self.rysujWykres:
          self.obiektFunkcji.narysujFunkcje(stop=najlepszaPozycjaRoju)
      return (i, najlepszaWartoscRoju, najlepszaPozycjaRoju)







class Osobnik:
  """
  Klasa osobnika roju
  """
  def __init__(self, predkoscMaksymalna, predkoscMinimalna, pozycjaMaksymalna, pozycjaMinimalna,delegat):
    # blad w obecnej pozycji
    self.obecnaWartosc = 0
    # najlepsza pozycja tego osobnika
    self.najlepszaPozycja = []
    # najelpszy blad tego osobnika
    self.najlepszaWartosc = -1
    # lista wszystkich pozycji tego osobnika
    self.listaPozycji = []
    # lista wszysktich predkosci tego osobnika
    self.listaPredkosci = []
    self.predkoscMaksymalna = predkoscMaksymalna
    self.predkoscMinimalna = predkoscMinimalna
    self.delegat = delegat
    self.pozycjaMaksymalna = pozycjaMaksymalna
    self.pozycjaMinimalna = pozycjaMinimalna
    # losujemy pozycje
    x1 = 20 * random.random()
    x2 = 20 * random.random()
    if random.random() > 0.5:
        x1 *= -1
    if random.random() > 0.5:
        x2 *= -1

    # losujemy predkosc
    v1 = random.random()
    v2 = random.random()
    if random.random() > 0.5:
        v1 *= -1
    if random.random() > 0.5:
        v2 *= -1

    self.najlepszaPozycja.append(0)
    self.listaPozycji.append([x1, x2])
    self.listaPredkosci.append([v1, v2])

  # przeliczenie wartosci w punkcie
  def wartoscOsobnika(self):
    # obliczamy wartosc funkcji w obecnej pozycji
    self.obecnaWartosc = self.funkcja(self.listaPozycji[-1])

    # jezeli to jest piersze obliczenie wartosci w pkt
    if self.najlepszaWartosc == -1 or self.obecnaWartosc < self.najlepszaWartosc:
      self.najlepszaWartosc = self.obecnaWartosc
      # uaktualniamy najnowsza pozycje
      self.najlepszaPozycja = self.listaPozycji[-1]
    return self.obecnaWartosc
  # obliczanie nowej predkosci
  def zmienPredkosc(self, najlepszaPozycjaRoju):
    """
      wyklad 5
    """
    # w decyduje o wplywie poprzedniej predkosci na nowa predkosc
    w = 1
    # c1 decduje o wplywie poprzedniej pozycji na nowa predkosc
    c1 = 2
    # c2 waga najlepszej pozycji calego roju
    c2 = 2
    # random ma rozklad normalny na [0,1]
    r1 = random.random()
    r2 = random.random()
    # nowa predkosc dla x
    vx1 = w * self.listaPredkosci[-1][0] + c1 * r1 * (self.najlepszaPozycja[0] - self.listaPozycji[-1][0]) + c2 * r2 * (najlepszaPozycjaRoju[0] - self.listaPozycji[-1][0])
    # nowa predkosc dla y
    vx2 = w * self.listaPredkosci[-1][1] + c1 * r1 * (self.najlepszaPozycja[1] - self.listaPozycji[-1][1]) + c2 * r2 * (najlepszaPozycjaRoju[1] - self.listaPozycji[-1][1])
    vx1 = self.ograniczPredkosc(vx1)
    vx2 = self.ograniczPredkosc(vx2)
    self.listaPredkosci.append([vx1, vx2])

  # funkcja ogranicza polozenie osobnikow
  def ograniczPozycje(self, pozycja):
      if pozycja > self.pozycjaMaksymalna:
        pozycja = self.pozycjaMaksymalna
      elif pozycja < self.pozycjaMinimalna:
        pozycja = self.pozycjaMinimalna
      return pozycja
  # funkcja ogranicza podana predkosc do maksimum
  def ograniczPredkosc(self, predkosc):
      if predkosc > self.predkoscMaksymalna:
        predkosc = self.predkoscMaksymalna
      elif predkosc < self.predkoscMinimalna:
        predkosc = self.predkoscMinimalna
      return predkosc
  # uaktualnienie pozycji na podstawie poprzedniej pozycji i predkosci
  def zmienPozycje(self):
    x1 = self.listaPozycji[-1][0] + self.listaPredkosci[-1][0]
    x2 = self.listaPozycji[-1][1] + self.listaPredkosci[-1][1]
    x1 = self.ograniczPozycje(x1)
    x2 = self.ograniczPozycje(x2)
    self.listaPozycji.append([x1, x2])

  def funkcja(self, x):
      return self.delegat.funkcja(x)


if __name__ == '__main__':
    fg = Goldstein()
    fe = Easom()
    fb = Beale()
    rb = Rosenbrock()
    fs = Sphere()
    srm = Swarm(obiektFunkcji=fe, iloscOsobnikowRoju=120, iloscIteracji=2000, rysujWykres=True)
    print srm.optymalizuj()
