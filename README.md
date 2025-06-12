## Code and data for Lorenz'63 and Drosophila GMN/ESN

### Generative Manifold Networks (GMN)
---
Requires python package [gmn](https://github.com/NonlinearDynamicsDSU/gmn), see [Documentation](https://nonlineardynamicsdsu.github.io/gmn/).

Requires python package [pyEDM](https://github.com/SugiharaLab/pyEDM), see [Documentation](https://sugiharalab.github.io/EDM_Documentation/).

---
### Lorenz'63

#### GMN

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

nx.draw(G,with_labels=True,font_size=16,
        node_color='lightgray',font_weight='bold')
plt.show()
```

---
![GMN_Network_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_Network_Lorenz3D.png)
---

Run GMN without a config file using the GMN application RunNoConfig.py. Parameters: E=3, tau=-7, taget node V3, GMN network file Lorenz3D_tgtV3_2.pkl, data file Lorenz3D_4k.csv. Start generation at index 2000, generate 1000 values.

```
./RunNoConfig.py -pS 2000 -pL 1000 -PT time -tn V3 \
-nf Lorenz3D_tgtV3_2.pkl -nd ../data/Lorenz3D_4k.csv -E 3 \
-tau -7 -o GMN_E3_tau-7_pS_2000_pL_1000.csv -PC V3 V1 V2 -P
```

---
![GMN_Generated_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_Generated_Lorenz3D.png)
---


Plot generated dynamics and RMSE with PlotGMN.py application.
```
./PlotGMN.py
```

---
![GMN_Generated_RMSE_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_Generated_RMSE_Lorenz3D.png)
---

#### ESN

The ESN class is contained in ESN/ESN.py. The application wrapper ESN/RunESN.py is a command line interface to run the ESN. 

Run ESN on Lorenz'63 with 1000, 2000, 3000 nodes. Train ESN on first 2000 points, generate 1000 points. 

```
cd ../ESN
./RunESN.py -R 1000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv \
-o ESN_R1000_Lorenz3D_pS_2000_pL_1000.csv -P

./RunESN.py -R 2000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv \
-o ESN_R2000_Lorenz3D_pS_2000_pL_1000.csv -P

./RunESN.py -R 3000 -t 1 2000 -e 2001 3000 -i ../data/Lorenz3D_4k.csv \
-o ESN_R3000_Lorenz3D_pS_2000_pL_1000.csv -P
```

---
![ESN_R1000_Generated_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/ESN_R1000_Generated_Lorenz3D.png)
---


Plot 1000 node generated dynamics and RMSE with PlotGMN.py application.
```
../GMN/PlotGMN.py -i ../ESN/ESN_R1000_Lorenz3D_pS_2000_pL_1000.csv \
-gv V1_ V2_ V3_ --title "ESN 1000"
```

---
![GMN_ESN_Generated_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_ESN_Generated_Lorenz3D.png)
---

#### Plot 3D dynamics
```
cd ..
```
```python
from pandas import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gmnLorenz = read_csv('GMN/GMN_E3_tau-7_pS_2000_pL_1000.csv')
esnR1000  = read_csv('ESN/ESN_R1000_Lorenz3D_pS_2000_pL_1000.csv')
esnR2000  = read_csv('ESN/ESN_R2000_Lorenz3D_pS_2000_pL_1000.csv')
esnR3000  = read_csv('ESN/ESN_R3000_Lorenz3D_pS_2000_pL_1000.csv')

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

ax1.set_title( 'GMN' )
ax2.set_title( 'ESN 1000' )
ax3.set_title( 'ESN 2000' )
ax4.set_title( 'ESN 3000' )

ax1.plot( gmnLorenz['V1'], gmnLorenz['V2'], gmnLorenz['V3'] )
ax2.plot( esnR1000['V1_'], esnR1000['V2_'], esnR1000['V3_'] )
ax3.plot( esnR2000['V1_'], esnR2000['V2_'], esnR2000['V3_'] )
ax4.plot( esnR3000['V1_'], esnR3000['V2_'], esnR3000['V3_'] )
plt.show()
```

---
![ESN_R1000_Generated_PlotGMN_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/ESN_R1000_Generated_PlotGMN_Lorenz3D.png)
---


#### Plot PSD

```python
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

gmnLorenz = read_csv('GMN/GMN_E3_tau-7_pS_2000_pL_1000.csv')
esnR1000  = read_csv('ESN/ESN_R1000_Lorenz3D_pS_2000_pL_1000.csv')
esnR2000  = read_csv('ESN/ESN_R2000_Lorenz3D_pS_2000_pL_1000.csv')
esnR3000  = read_csv('ESN/ESN_R3000_Lorenz3D_pS_2000_pL_1000.csv')

nFFT    = 300
win     = mlab.window_hanning # mlab.window_none # mlab.window_hanning
overlap = 50
deltaT  = (gmnLorenz.loc[1,'Time'] - gmnLorenz.loc[0,'Time']).round(5)
fs      = 1 / deltaT

fig0, ax0 = plt.subplots(1, 1)
ax0.psd( esnR1000.loc[50:,'V1'], NFFT = nFFT, Fs = fs,
         window = win, noverlap = overlap, label = 'V1', lw = 3 )
ax0.psd( gmnLorenz.loc[100:,'V1'], NFFT = nFFT, Fs = fs,
         window = win, noverlap = overlap, label = 'GMN', lw = 2 )
ax0.psd( esnR1000.loc[50:,'V1_'], NFFT = nFFT, Fs = fs,
         window = win, noverlap = overlap, label = 'ESN 1k', lw = 2 )
ax0.psd( esnR2000.loc[50:,'V1_'], NFFT = nFFT, Fs = fs,
         window = win, noverlap = overlap, label = 'ESN 2k', lw = 2 )
ax0.psd( esnR3000.loc[50:,'V1_'], NFFT = nFFT, Fs = fs,
         window = win, noverlap = overlap, label = 'ESN 3k', lw = 2 )
ax0.legend()
plt.show()
```

---
![GMN_ESN_PSD_Lorenz3D](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_ESN_PSD_Lorenz3D.png)
---


---
### Drosophila
---

---
#### GMN

Compute interaction matrix with GMN InteractionMatrix.py application 
```
GMN/InteractionMatrix.py -d data/Fly80XY_norm_1061.csv -rhoDiff -P -E 7
```

Plot Fly GMN Network
```
cd GMN
```
```python
from pickle import load
import matplotlib.pyplot as plt
import networkx as nx

with open('Fly80_norm_1061_rhoDiff_Network_D4.pkl', 'rb') as f:
     network = load(f)

print( network['Map'] )

G = network['Graph']

nx.draw_kamada_kawai(G,with_labels=True,alpha=0.7,font_size=16,
                     node_color='lightgray',font_weight='bold')
plt.show()
```

---
![GMN_Network_Drosophila](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_Network_Drosophila.png)
---


Run GMN on target node FWD with Tp 1 -E 7 -tau -8
```
./RunNoConfig.py -pS 580 -pL 480 -PT index -tn FWD \
-nf Fly80_norm_1061_rhoDiff_Network_D4.pkl \
-nd ../data/Fly80XY_norm_1061.csv -PT time -nn "Fly 80 : FWD" \
-Tp 1 -E 7 -tau -8 -do GMN_Fly80_1061_rhoDiff_D4_E7_tau-8.csv \
-PC FWD TS2 TS30 -P
```

---
![GMN_Generated_Drosophila](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_Generated_Drosophila.png)
---


---
#### ESN

3000 node ESN to generate FWD : Same TS input as GMN with network Fly80_norm_rhoDiff_Network_D4.pkl

```
cd ../ESN
```
```
./RunESN.py -i ../data/Fly80XY_norm_1061.csv -sr 0.9 -lr 0.5 -time index \
-iv TS1 TS2 TS3 TS4 TS5 TS6 TS7 TS8 TS9 TS10 TS11 TS12 TS13 TS14 TS15 TS16 \
TS17 TS18 TS19 TS20 TS21 TS22 TS23 TS24 TS25 TS26 TS27 TS28 TS29 TS30 TS31 \
TS32 TS33 TS34 TS35 TS36 TS37 TS38 TS39 TS40 TS41 TS42 TS43 TS44 TS45 TS46 \
TS47 TS48 TS49 TS50 TS51 TS52 TS53 TS54 TS55 TS56 TS57 TS58 TS59 TS60 TS61 \
TS62 TS63 TS64 TS65 TS66 TS67 TS68 TS69 TS70 TS71 TS72 TS73 TS74 TS75 TS76 \
TS77 TS78 TS79 TS80 FWD -t 1 500 -b 100 -e 601 1000 -R 3000 \
-o ESN_Fly80_1061.csv -P
```

---
![ESN_Generated_Drosophila](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/ESN_Generated_Drosophila.png)
---


#### Plot results
```
cd ..
```
```python
from pandas import read_csv
import matplotlib.pyplot as plt

gmn = read_csv('GMN/GMN_Fly80_1061_rhoDiff_D4_E7_tau-8.csv').iloc[:420,:]
esn = read_csv('ESN/ESN_Fly80_1061.csv')

ax = esn.plot('index','FWD',lw=2)   # FWD data
gmn.plot('index','FWD',ax=ax,lw=2,label='GMN')  # GMN 
esn.plot('index','FWD_',ax=ax,lw=2,label='ESN') # ESN
plt.show()
```

---
![GMN_ESN_DrosophilaFWD_Generate](https://raw.githubusercontent.com/pao-unit/GMN_ESN_Examples/main/plots/GMN_ESN_DrosophilaFWD_Generate.png)
---

