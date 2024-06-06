## {{{ imports
import importlib

import plotter as p

importlib.reload(p)

import os

## }}}

##{{{ fig1 de métodos de tesis

importlib.reload(p)

figs = (
    "/home/ttdduu/lsd/tesislab/extra_files/la_tesis/resultados/figs/descriptiva/finales"
)
col1 = [f"{figs}/{i}" for i in os.listdir(figs) if "sonograma" in i]
col2 = [f"{figs}/{i}" for i in os.listdir(figs) if "solo" in i]

filas = [[col1[fila_number], col2[fila_number]] for fila_number in [0, 1, 2]]

p.plot_imgs_3x2(
    filas[0], filas[1], filas[2], ax_titles=["cantos de hembra", "cantos de macho"]
)
##}}}

## {{{

p.p("/home/ttdduu/lsd/tesislab/datos/pitches/beta/B/song1/25.txt")
# p.pdir('/home/ttdduu/lsd/tesislab/datos/pitches/beta/A/song1')
## }}}

## {{{ repertorio beta

lista_pngs = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/machoHA.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/machoHB.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/macho19.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/macho23.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/macho34.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/machoHEC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/machoHAC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pngs/machoHAC2.png",
]
lista_dirs_pitches = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/A",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/B",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/19",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/23",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/34",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/HEC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/HAC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_machos/pitches/HAC2",
]


p.plot_superpuestos(lista_dirs_pitches, lista_pngs)

## }}}

## {{{ repertorio beta

lista_pngs = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/betaHA.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/betaHB.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/beta19.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/beta23.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/beta34.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/betaHEC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/betaHAC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pngs/betaHAC2.png",
]
lista_dirs_pitches = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/A",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/B",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/19",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/23",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/34",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/HEC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/HAC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/repertorio_beta/pitches/HAC2",
]


p.plot_superpuestos(lista_dirs_pitches, lista_pngs)

## }}}

## {{{ repertorio beta

lista_pngs = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/HA.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/HB.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/H19.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/H23.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/34.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/HEC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/HAC1.png",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pngs/HAC2.png",
]
lista_dirs_pitches = [
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/A",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/B",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/19",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/23",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/34",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/HEC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/HAC1",
    "/home/ttdduu/lsd/tesislab/datos/pitches_no_para_red/plot_repertorio/pitches/HAC2",
]


p.plot_superpuestos(lista_dirs_pitches, lista_pngs)

## }}}

## {{{

importlib.reload(p)

fila1 = [
    "/home/ttdduu/lsd/tesislab/extra_files/la_tesis/resultados/figs/descriptiva/alfa_rg_din.png",
    "/home/ttdduu/lsd/tesislab/extra_files/la_tesis/resultados/figs/descriptiva/beta_rgdin.png",
]
fila2 = [
    "/home/ttdduu/lsd/tesislab/extra_files/la_tesis/resultados/figs/descriptiva/alfa_rgdin2.png",
    "/home/ttdduu/lsd/tesislab/extra_files/la_tesis/resultados/figs/descriptiva/beta_rgdin2.png",
]
tiempo_beta = list(np.round(np.arange(0, 0.867, 0.2), 2))
tiempo_alfa = list(np.round(np.arange(0, 1.574, 0.35), 2))
frec = [int(i) for i in list(np.linspace(1000, 7000, 7))]
p.plot_imgs_2x2(fila1, fila2, tiempo_alfa, tiempo_beta, frec)

## }}}

## {{{ histogramas de pitches
importlib.reload(p)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/home/ttdduu/lsd/tesislab/datos/pitches/violinplot/datos_crudos_violinplot.csv"
)
# Define the individuals and tipos
individuals = ["A", "HAC2", "19", "23"]
tipos = ["alfa", "beta", "macho"]
p.histogramas_pitches(df, tipos=tipos, individuals=individuals)
## }}}
