## Code and data for Lorenz'63 and Drosophila GMN/ESN

### Generative Manifold Networks (GMN)
---
Requires python package [gmn](https://github.com/NonlinearDynamicsDSU/gmn), see [Documentation](https://nonlineardynamicsdsu.github.io/gmn/).

Requires python package [pyEDM](https://github.com/SugiharaLab/pyEDM), see [Documentation](https://sugiharalab.github.io/EDM_Documentation/).

---
#### Lorenz'63

Examine GMN network:
```
cd GMN
```
```python
from pickle import load

with open('Lorenz3D_tgtV3_2.pkl', 'rb') as f:
     network = load(f)

network['Map']
### {'V3': ['V1', 'V2'], 'V1': [], 'V2': ['V1']}
network['Graph']
### <networkx.classes.digraph.DiGraph object at 0x...>

# Plot using networkx
import matplotlib.pyplot as plt
import networkx as nx

G = network['Graph']

nx.draw_kamada_kawai(G,with_labels=True,alpha=0.7,font_size=16,node_color='lightgray',font_weight='bold')
plt.show()
```

Run GMN without a config file using parameters -E 4 -tau -7, GMN network file Lorenz3D_tgtV3_2.pkl. Start generation at index 2000, generate 1000 values.

```
./RunNoConfig.py -pS 2000 -pL 1000 -PT time -PC V3 V1 V2 -tn V3 -nf Lorenz3D_tgtV3_2.pkl -nd ../data/Lorenz3D_4k.csv -E 4 -tau -7 -o GMN_E4_tau-7_pS_2000_pL_1000.csv -P
```

---
#### Drosophila

Plot Fly GMN Network
```python
from pickle import load
import matplotlib.pyplot as plt
import networkx as nx

with open('Fly80_norm_1061_rhoDiff_Network_D4.pkl', 'rb') as f:
     network = load(f)

network['Map']
G = network['Graph']

nx.draw_kamada_kawai(G,with_labels=True,alpha=0.7,font_size=16,node_color='lightgray',font_weight='bold')
plt.show()
```

Run GMN with Tp 1 -E 7 -tau -8
```
./RunNoConfig.py -pS 580 -pL 480 -PT index -PC FWD Left_Right TS2 TS30 -tn FWD -nf Fly80_norm_1061_rhoDiff_Network_D4.pkl -nd ../data/Fly80XY_norm_1061.csv -PT time -nn "Fly 80 : Left-Right FWD" -Tp 1 -E 7 -tau -8 -do GMN_Fly80_1061_rhoDiff_D4_E7_tau-8.csv -P
```

---
### Echo State Network (ESN)
---

#### Lorenz'63 

Run ESN on Lorenz'63 with 1000, 2000, 3000 nodes. Train ESN on first 2000 points, generate 1000 points. 

```
cd ESN
./RunESN.py -R 1000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv -o ESN_R1000_Lorenz3D_pS_2000_pL_1000.csv -P
./RunESN.py -R 2000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv -o ESN_R2000_Lorenz3D_pS_2000_pL_1000.csv -P
./RunESN.py -R 3000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv -o ESN_R3000_Lorenz3D_pS_2000_pL_1000.csv -P
```

---
#### Drosophila

3000 node ESN to generate FWD : Same TS input as GMN with network Fly80_norm_rhoDiff_Network_D4.pkl

```
./RunESN.py -i ../data/Fly80XY_norm_1061.csv -sr 0.9 -lr 0.5 -time index -iv TS1 TS2 TS3 TS4 TS5 TS6 TS7 TS8 TS9 TS10 TS11 TS12 TS13 TS14 TS15 TS16 TS17 TS18 TS19 TS20 TS21 TS22 TS23 TS24 TS25 TS26 TS27 TS28 TS29 TS30 TS31 TS32 TS33 TS34 TS35 TS36 TS37 TS38 TS39 TS40 TS41 TS42 TS43 TS44 TS45 TS46 TS47 TS48 TS49 TS50 TS51 TS52 TS53 TS54 TS55 TS56 TS57 TS58 TS59 TS60 TS61 TS62 TS63 TS64 TS65 TS66 TS67 TS68 TS69 TS70 TS71 TS72 TS73 TS74 TS75 TS76 TS77 TS78 TS79 TS80 FWD -t 1 500 -b 100 -e 601 1000 -R 3000 -o ESN_Fly80_1061_rhoDiff_D4.csv -P
```
