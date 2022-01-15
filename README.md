# Prognozowanie modelem SARIMA

Kod programu znajduje się w folderze `src`. Folder `docs` zawiera sprawozdanie oraz jego kod źródłowy.

## Przygotowanie środowiska

Program wymaga pythona w wersji `3.x`. Jeżeli system jako domyślną wersję używa pythona w wersji `2.x` należy zastąpić `python` bezpośrednią nazwą (zwykle `python3`).

W celu zainstalowania wymaganych pakietów należy uruchomić

```CMD
pip install -r requirements.txt
```

## Uruchamianie

W celu uruchomienia programu w trybie prognozowania należy uruchomić plik `forecast.py` znajdując się w folderze `src`.

```CMD
python forecast.py
```

Wynikiem programu są okna z wykresami przedstawiającymi wyniki prognozowania. Konsola zawiera błędy otrzymanych wyników.

Jeżeli plik z danymi nie znajduje się w aktualnym folderze, lub ma inną nazwę, należy przekazać ścieżkę do niego jako parametr `i`

```CMD
python forecast.py -i <pathToData>
```

Aby uruchomić aplikację w trybie analizy należy dodać parametr `a`. Aplikacja wyświetla wykresy użyte podczas analizy danych i kończy działanie.

```CMD
python forecast.py -a
```
