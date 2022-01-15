Kod programu znajduje się w folderze `src`. Folder `docs` zawiera sprawozdanie oraz jego kod źródłowy.

W celu zainstalowania wymaganych pakietów należy uruchomić

```CMD
pip install -r requirements.txt
```

W celu uruchomienia programu w trybie prognozowania należy uruchomić plik `forecast.py` znajdując się w folderze `src`.

```CMD
python forecast.py
```

Wynikiem programu będą okna z wykresami przedstawiającymi wyniki prognozowania oraz wypis na konsoli zawierający błędy otrzymanych wyników.

Jeżeli plik z danymi nie znajduje się w aktualnym folderze, lub ma inną nazwę, należy przekazać ścieżkę do niego jako parametr `i`

```CMD
python forecast.py -i <pathToData>
```

Aby uruchomić aplikację w trybie analizy należy dodać parametr `a`. Aplikacja wyświetla wykresy użyte podczas analizy danych i kończy działanie.

```CMD
python forecast.py -a
```
