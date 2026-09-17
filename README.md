# Poređenje neuronskih mreža na EuroSAT skupu

Ovaj projekat poredi nekoliko neuronskih mreža za klasifikaciju satelitskih slika iz EuroSAT skupa podataka. Glavni cilj je poređenje DPN modela sa ResNet i DenseNet arhitekturama, dok je mali Vision Transformer dodat kao još jedan eksperiment.

Modeli su implementirani od nule u PyTorchu:

- ResNet18
- DenseNet121
- Dual Path Network (DPN)
- Vision Transformer (ViT)

## Skup podataka

EuroSAT sadrži 27.000 RGB slika dimenzija 64 x 64 piksela i ukupno 10 klasa. Podaci su podeljeni na:

- trening skup: 16.200 slika
- validacioni skup: 5.400 slika
- test skup: 5.400 slika

Za trening slike korišćeno je nasumično horizontalno i vertikalno okretanje. Validacione i test slike nisu menjane, jer služe za proveru rada modela.

## Struktura projekta

```text
DPN_EuroSAT/
|-- data/
|   |-- raw/
|   `-- splits/
|-- notebooks/
|   |-- 01_eurosat_analysis.ipynb
|   `-- 02_train_final.ipynb
|-- results/
|   |-- checkpoints/
|   |-- figures/
|   `-- metrics/
|-- src/
|   `-- models/
|       |-- resnet.py
|       |-- densenet.py
|       |-- dpn.py
|       `-- vit.py
`-- tests/
```

## Treniranje

Svi modeli su trenirani 10 epoha. Korišćen je Adam optimizator, learning rate 0.001 i batch veličina 64. Funkcija greške je CrossEntropyLoss.

Treniranje je izvršeno na Google Colab Tesla T4 grafičkoj kartici, pošto bi na lokalnom računaru bez CUDA podrške trajalo dosta duže. Notebook je ipak prilagođen tako da može da radi i lokalno na procesoru.

## Rezultati

| Model | Najbolja validaciona tačnost | Test tačnost |
|---|---:|---:|
| ResNet18 | 87.04% | 87.56% |
| DenseNet121 | 91.44% | 91.81% |
| DPN | 90.81% | 91.67% |
| ViT | 86.20% | 86.37% |

Najbolju test tačnost ostvario je DenseNet121. DPN je ostvario skoro isti rezultat, sa razlikom od samo 0.14 procentnih poena. U odnosu na ResNet18, DPN je bio bolji za 4.11 procentnih poena. ViT je imao najmanje parametara, ali i nešto slabiju tačnost.

## Pokretanje projekta

Nakon kreiranja i aktiviranja virtuelnog okruženja potrebno je instalirati biblioteke:

```powershell
python -m pip install torch torchvision matplotlib pandas scikit-learn jupyter
```

Zatim se redom otvaraju notebook fajlovi iz foldera `notebooks`. Prvi notebook služi za pregled podataka, a drugi sadrži treniranje, testiranje i poređenje modela.

Sačuvani rezultati, grafikoni i istrenirani modeli nalaze se u folderu `results`.
