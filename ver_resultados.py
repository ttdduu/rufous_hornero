## {{{ imports
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import load_weights
import plotter as p
import metricas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import importlib
import os
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
import random
from sklearn.metrics import silhouette_score

try:
    from spherecluster import SphericalKMeans
except ImportError:
    pass
import importlib


# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print(physical_devices)  # nice

# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
max_clusters = 14

## }}}

tipo = "beta"
## {{{ listas de weights (definir el tipo de sílaba)
# {{{ tipo = alfa

if tipo == "alfa":
    trains_nulo = [i for i in range(153, 173)]
    # trains = [i for i in range(227, 277)] # los de 20 epochs
    trains = [
        # 273,
        # 276,
        # 277,
        # 278,
        # 281,
        # 282,
        # 283,
        # 286,
        #
        288,
        290,
        292,
        293,
        295,
        297,
        299,
        300,
        301,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        312,
        313,
        314,
    ]

    lista_weights_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        )
    ]
    lista_test_pkl_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        )
    ]
    lista_weights = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]
    lista_test_pkl = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]
# }}}
# {{{ tipo = beta

if tipo == "beta":
    trains_nulo = [i for i in range(33, 62)]  # para los beta nulos
    trains = [i for i in range(1, 33)] + [
        i for i in range(64, 114)
    ]  # hasta 32 son no-nulos de beta

    lista_weights_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]
    lista_test_pkl_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]

    lista_weights = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]
    lista_test_pkl = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]
    trains_beta_chotos = [i - 1 for i in [44, 40, 39, 36, 24, 19, 18, 11, 8, 34, 31]]
    lista_weights = [
        lista_weights[i]
        for i in range(len(lista_weights))
        if i not in trains_beta_chotos
    ]
    lista_test_pkl = [
        lista_test_pkl[i]
        for i in range(len(lista_test_pkl))
        if i not in trains_beta_chotos
    ]

# }}}
# {{{ tipo = machos

if tipo == "machos":
    trains = [i for i in range(10, 32)] + [i for i in range(62, 112)]
    # trains = [i for i in range(70,80)]
    trains_nulo = [i for i in range(34, 52)]
    lista_weights_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        )
    ]
    lista_test_pkl_nulo = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains_nulo
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_networknulo.h5"
        )
    ]
    lista_weights = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]

    lista_test_pkl = [
        f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/test_set.pkl"
        for i in trains
        if os.path.exists(
            f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{i}/base_network.h5"
        )
    ]

    trains_machos_chotos = [
        i - 1
        for i in [
            7,
            11,
            28,
            34,
            59,
            61,
            60,
            58,
            6,
            33,
            27,
            10,
        ]
    ]
    lista_weights = [
        lista_weights[i]
        for i in range(len(lista_weights))
        if i not in trains_machos_chotos
    ]
    lista_test_pkl = [
        lista_test_pkl[i]
        for i in range(len(lista_test_pkl))
        if i not in trains_machos_chotos
    ]
# }}}

## }}}

# random_integers = [random.randint(0, 78) for _ in range(60)]
# lista_weights, lista_test_pkl = [lista_weights[i] for i in random_integers],[lista_test_pkl[i] for i in random_integers]
#
## {{{ ver varios coefs de silhouette nulos vs reales y hacer dataframes

# {{{ calcular sils y ver plot

importlib.reload(p)
importlib.reload(metricas)

# {{{ ) = p.plot_silhouette_nulo_vs_real(
(
    lista_kmeans_score_real,
    lista_real_score_real,
    lista_kmeans_score_nulo,
    lista_real_score_nulo,
    promedios_score_k,
    sd_score_k,
    promedios_score_k_nulo,
    sd_score_k_nulo,
    fig,  # la figura
    ax,
) = p.plot_silhouette_nulo_vs_real(
    lista_weights,
    lista_weights_nulo,
    lista_test_pkl,
    lista_test_pkl_nulo,
    puntos_reales=True,
)

# }}}
# }}}

# {{{ reasignar nombres de vars

a = lista_kmeans_score_real  # no es lo que termino usando
k8 = [lista[6] for lista in a]
b = lista_real_score_real  # no es lo que termino usando
r8 = b
resta_k8_r8 = [j - r8[i] for i, j in enumerate(k8)]
media_scores_reales = np.mean(b)
c = lista_kmeans_score_nulo  # no es lo que termino usando
d = lista_real_score_nulo  # no es lo que termino usando
media_scores_nulos = np.mean(d)
e = promedios_score_k
f = sd_score_k
g = promedios_score_k_nulo
h = sd_score_k_nulo
listas_silhouette_ordenadas_por_k = [[j[k] for i, j in enumerate(a)] for k in range(13)]
i = listas_silhouette_ordenadas_por_k
pruebas_KS_vs_k8 = [
    ks_2samp(k, i[6]) for k in i
]  # para el modelo real, si los sil de los clusters de cada K provienen de misma dist que los de k=8.
pruebas_KS_vs_k8_reales = [
    ks_2samp(b, k) for k in i
]  # para el modelo real, si los sil de los clusters posta provienen de misma dist que los de kmeans.
# }}}

# {{{ generar dataframes

fila_ks = [f"{j.statistic:.2e}; {j.pvalue:.2e}" for j in pruebas_KS_vs_k8]
fila_ks_reales = [f"{j.statistic:.2e}; {j.pvalue:.2e}" for j in pruebas_KS_vs_k8_reales]
e_f = [
    f"{np.round(float(e[i]),decimals=2)}\u00B1{np.round(float(f[i]),decimals=2)}"
    for i in range(len(e))
]
g_h = [
    f"{np.round(float(g[i]),decimals=2)}\u00B1{np.round(float(h[i]),decimals=2)}"
    for i in range(len(g))
]
fila_score_real = [
    "-" if i != 6 else media_scores_reales for i in range(max_clusters - 1)
]
fila_score_nulo = [
    "-" if i != 6 else media_scores_nulos for i in range(max_clusters - 1)
]

df = pd.DataFrame(
    [e_f, fila_ks, fila_score_real, g_h, fila_ks_reales, fila_score_nulo],
    columns=[f"k={i}" for i in range(2, max_clusters + 1)],
).round(2)

titulo = f"Coeficientes de silhouette y tests de significancia de modelos reales (n={len(lista_weights)} y nulos (n={len(lista_weights_nulo)})"
df.columns = pd.MultiIndex.from_tuples(
    zip([titulo] + [""] * (max_clusters - 2), df.columns)
)

df.index = [
    f"promedios score K con sd",
    "estadístico y pval de test KS de cada k vs k=8",
    "score de clusters reales, modelo real",
    "promedios score K nulo con sd",
    "estadístico y pval de test KS de cada k vs clusters reales",
    "score de clusters reales, modelo nulo",
]

df_crudos = pd.DataFrame(
    i,
    columns=[f"train {i}" for i in range(1, len(i[0]) + 1)],
    index=[f"k={i}" for i in range(2, max_clusters + 1)],
)

path_metricas = (
    f"/home/ttdduu/lsd/tesislab/entrenamientos/metricas/sils/{tipo}/sils_R_agrego_EE"
)
counter = 1
while os.path.exists(f"{path_metricas}/medias_silhouette_ks-{counter}.csv"):
    counter += 1
counter -= 1
df.to_csv(f"{path_metricas}/medias_silhouette_ks-{counter}.csv")
df_crudos.to_csv(f"{path_metricas}/datos_crudos_silhouette-{counter}.csv")
fig.savefig(f"b.png", dpi=600)

import dunnett

importlib.reload(dunnett)
dunnett.generar_datos(tipo, counter)

## }}}


## }}}

"""
resultados para una sola predicción
"""
## {{{ predicts = base_network.predict(test_set)

n_train = 23
importlib.reload(p)
importlib.reload(metricas)

path_weights, test_pkl = (
    f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{n_train}/base_network.h5",
    f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{n_train}/test_set.pkl",
)

test_set, test_tensors_belong_to, base_network = load_weights.load(
    path_weights, test_pkl
)

predicts = base_network.predict(test_set)

axtitles = [
    "clusters reales",
    "clusters de kmeans con sus centroides en negro",
    "centroides reales y de kmeans",
]
#

## }}}
## {{{ p.prediction_plots(
importlib.reload(p)
p.prediction_plots(
    test_set=test_set,
    test_tensors_belong_to=test_tensors_belong_to,
    base_network=base_network,
    solo_centroides=False,
)

# sil1 = p.plotear_silhouettes([path_weights], [test_pkl], spherical=True)[0].figure

p.comparar_red_kmeans(
    test_set,  # por default el test set de una sesión interactiva actual
    test_tensors_belong_to,  # same
    base_network,
    axtitles,
    spherical=True,
    completo=False,
)


## }}}
## {{{ p.comparar_red_kmeans(


# comparar los k en una predicción
#
# lista_k = [4,8,11,15]
# p.plot_distintos_k(predicts,lista_k,spherical=True,)
# p.plot_distintos_k(predicts,lista_k,spherical=False,)

# test_tensors_belong_to = [i[: i.find("_")] for i in test_tensors_belong_to]
importlib.reload(p)
a, b = p.comparar_red_kmeans(
    test_set, test_tensors_belong_to, base_network, axtitles, spherical=True
)

metricas.sum_of_distances(a)
metricas.sum_of_distances(b)

## }}}

## {{{ ver todas las preds


importlib.reload(p)
importlib.reload(metricas)
for i, j in enumerate(lista_weights):
    path_weights, test_pkl = (j, lista_test_pkl[i])

    test_set, test_tensors_belong_to, base_network = load_weights.load(
        path_weights, test_pkl
    )

    predicts = base_network.predict(test_set)

    axtitles = [
        "clusters reales",
        "clusters de kmeans con sus centroides en negro",
        "centroides reales y kmeans",
    ]

    p.prediction_plots(
        test_set=test_set,
        test_tensors_belong_to=test_tensors_belong_to,
        base_network=base_network,
        solo_centroides=False,
    )

## }}}

"""
para hacer heatmap
"""
## {{{ predicts = base_network.predict(test_set)

tipo = "beta"
# n_train = 300 # alfa
n_train = 101  # beta
# n_train = 11 # macho
importlib.reload(p)
importlib.reload(metricas)

path_weights, test_pkl = (
    f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{n_train}/base_network.h5",
    f"/home/ttdduu/lsd/tesislab/entrenamientos/{tipo}/{n_train}/test_set.pkl",
)

test_set, test_tensors_belong_to, base_network = load_weights.load(
    path_weights, test_pkl
)

predicts = base_network.predict(test_set)

axtitles = [
    "clusters reales",
    "clusters de kmeans con sus centroides en negro",
    "centroides reales y kmeans",
]
#

## }}}
## {{{ centroides_k, centroides_red = p.comparar_red_kmeans(

k = 7
n = k

importlib.reload(p)
centroides_k, centroides_red = p.comparar_red_kmeans(
    test_set, test_tensors_belong_to, base_network, axtitles, spherical=True, k=k
)
centros = np.vstack([np.array(centroides_k), centroides_red])

## }}}
diagonal = False  # truchísmo per va
if diagonal:
    # {{{ from sklearn.metrics.pairwise import cosine_similarity --> dataframe

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim_matrix = cosine_similarity(centros)
    # display(cosine_sim_matrix)
    cosine_sim_matrix_red = cosine_sim_matrix[n:]
    # display(cosine_sim_matrix_red)
    second_lowest_indices = np.argsort(cosine_sim_matrix_red, axis=1)[:, -2]
    distancias_red_kmeans_cercanas = [
        fila[second_lowest_indices[i]] for i, fila in enumerate(cosine_sim_matrix_red)
    ]

    # dict_promedios
    # second_lowest_indices

    # distancias_red_kmeans_cercanas

    df = pd.DataFrame(
        np.hstack(
            [cosine_sim_matrix_red[:, n:], cosine_sim_matrix_red[:, :n]]
        )  # REVISAR
    )

    # Customize row and column labels
    row_labels = [
        "r{}".format(i) for i in range(1, n + 1)
    ]  # Row labels r1, r2, ..., r8
    df.index = row_labels

    if "nulo" not in path_weights:
        column_labels = row_labels + [
            "k{}".format(i)
            for i in range(1, n + 1)  # para tener el orden que me da la diagonal
            # "k{}".format(i) for i in second_lowest_indices --> no lo voy a usar. los nombres originales de los clusters son aleatorios. voy a renombrar los clusters.
        ]  # Column labels k1, k2, ..., k8, r1, r2, ..., r8

        # Get the names of the last 8 columns
        last_8_cols = df.columns[-n:]

        # Reorder the last 8 columns according to the specified order
        reordered_cols = [last_8_cols[i] for i in second_lowest_indices]

        # Reorder the DataFrame columns
        df = df[list(df.columns[:-n]) + reordered_cols]
        df.columns = column_labels

    else:
        # if tipo=="beta":
        column_labels = row_labels + [
            "k{}".format(i) for i in second_lowest_indices
        ]  # Column labels k1, k2, ..., k8, r1, r2, ..., r8
        # Create a list to store column names based on the reordering logic
        new_order = list(df.columns)

        for i, idx in enumerate(second_lowest_indices):
            if idx > 7:
                print(4554)
                new_order[n + i] = list(df.columns)[idx]
        # Reorder the DataFrame
        df = df[new_order]
        # Customize row and column labels
        row_labels = [
            "r{}".format(i) for i in range(1, n + 1)
        ]  # Row labels r1, r2, ..., r8
        df.index = row_labels
        column_labels = row_labels + [
            "k{}".format(i)
            for i in range(1, n + 1)  # para tener el orden que me da la diagonal
            # "k{}".format(i) for i in second_lowest_indices --> no lo voy a usar. los nombres originales de los clusters son aleatorios. voy a renombrar los clusters.
        ]  # Column labels k1, k2, ..., k8, r1, r2, ..., r8
        df.columns = column_labels

    # }}}
else:
    # {{{ from sklearn.metrics.pairwise again

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim_matrix = cosine_similarity(centros)
    # display(cosine_sim_matrix)
    cosine_sim_matrix_red = cosine_sim_matrix[n:]
    # display(cosine_sim_matrix_red)
    second_lowest_indices = np.argsort(cosine_sim_matrix_red, axis=1)[:, -2]

    df = pd.DataFrame(
        np.hstack(
            [cosine_sim_matrix_red[:, n:], cosine_sim_matrix_red[:, :n]]
        )  # REVISAR
    )

    # Customize row and column labels
    row_labels = [
        "r{}".format(i) for i in range(1, n + 1)
    ]  # Row labels r1, r2, ..., r8
    df.index = row_labels

    column_labels = row_labels + ["k{}".format(i) for i in range(1, n + 1)]

    df.columns = column_labels

    # }}}
## {{{ sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1)

plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1)
plt.title("Cosine Similarity Heatmap")
plt.xlabel("Clusters reales y de K-Means")
plt.ylabel("Clusters reales")
cell_height = 1
cell_width = 1
llamados = []

# for para marcar los max si labeleo k1-k8
# for index, pos in enumerate(second_lowest_indices):
# if pos > 7:
# pos = pos - 16
# highlight_position = [index, pos + 8]
# highlight_rectangle = plt.Rectangle(
# (highlight_position[1], highlight_position[0]),
# cell_width,
# cell_height,
# fill=False,
# edgecolor="black",
# lw=4,
# )
# plt.gca().add_patch(highlight_rectangle)

if "nulo" not in path_weights:
    for i in range(n):
        highlight_rectangle = plt.Rectangle(
            (i + n, i),
            cell_width,
            cell_height,
            fill=False,
            edgecolor="black",
            lw=4,
        )
        plt.gca().add_patch(highlight_rectangle)
else:
    for i, idx in enumerate(second_lowest_indices):
        if idx > n:
            highlight_rectangle = plt.Rectangle(
                (idx - (n), i),
                cell_width,
                cell_height,
                fill=False,
                edgecolor="black",
                lw=4,
            )
            plt.gca().add_patch(highlight_rectangle)
        else:
            highlight_rectangle = plt.Rectangle(
                (idx + (n), i),
                cell_width,
                cell_height,
                fill=False,
                edgecolor="black",
                lw=4,
            )
            plt.gca().add_patch(highlight_rectangle)


# Add inferior titles
plt.axvline(x=n, color="black", linestyle="-", lw=4)
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.show()

## }}}

"""
índice de estructura de a pares para un heatmap

func que vaya fila por fila. en cada fila R encuentre el cluster r y el cluster k más
cercanos. calculo dos ángulos:
- el angulo entre el origen y los puntos R-k
- el ángulo entre el origen y los puntos R-r.

alfa_Rk debería ser chico, y el alfa_Rr grande. entonces calculo alfa_Rk/alfa_Rr que debería ser chico. 1 si el k y el r más cercanos están en el mismo lugar, >1 si el
cluster más cercano es un r en vez de un k. este nro debería tender a 0 cuanto mejor sea
el clustering.
"""
## {{{ generar predicciones

# {{{ def
from sklearn.metrics.pairwise import cosine_similarity


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


importlib.reload(metricas)


def para_todas_las_preds(predicciones, dict_medida_calidad, individuals):
    for i in predicciones.keys():
        km = KMeans(n_clusters=8)
        km.fit(
            predicciones[i][0]
        )  # suponiendo que i es una lista que en 1er elemento tiene las preds y en el 2do test_tensors_belong_to que para Kmeans no lo uso
        centros_k = km.cluster_centers_
        class_labels = individuals
        centros_R = metricas.centros_de_masa(
            class_labels, predicciones[i][1], predicciones[i][0]
        )

        suma_distances_r = metricas.sum_distances(centros_R)

        nan_rows = np.where(np.isnan(centros_R).any(axis=1))[0]

        if nan_rows.any():
            # print(centros_R)
            centros_R = np.delete(centros_R, nan_rows, axis=0)
            centros_k = np.delete(centros_R, nan_rows, axis=0)
        # calculo el coef de silhouette de estas preds de clusters reales

        # sil_real = silhouette_score(
        # predicciones[i][0], predicciones[i][1], metric="cosine"
        # )
        sil_real = None

        medida_de_calidad = mas_cercanos(
            centros_R, centros_k, suma_distances_r, sil_real
        )
        dict_medida_calidad[i] = medida_de_calidad
    # print(dict_medida_calidad)

    return dict_medida_calidad


def mas_cercanos(
    centros_r, centros_k, suma_distances_r, sil_real=None
):  # lo llamo una vez para cada set de predicciones
    cocientes_de_angulos_de_cada_R = []
    # solo_cocs = []
    # suma_de_cada_pred = []

    # {{{ lo que hice mil veces
    for idx, R in enumerate(centros_r):
        # if suma_distances_r > 10:
        centros = centros_r
        centros = np.delete(centros, idx, axis=0)
        similitudes_r = cosine_similarity([R], centros)
        indice_mas_cercano_r = np.argmax(similitudes_r)
        similitudes_k = cosine_similarity([R], centros_k)
        indice_mas_cercano_k = np.argmax(similitudes_k)

        angulo_Rr = similitudes_r[0][indice_mas_cercano_r]
        angulo_Rk = similitudes_k[0][indice_mas_cercano_k]

        cociente = angulo_Rk / angulo_Rr  # en realidad es similaridad!
        # print(f'Rr: {angulo_Rr}, Rk: {angulo_Rk}, cociente')
        cociente_norm = cociente
        # cociente_norm = suma_distances_r  # DONE

        cocientes_de_angulos_de_cada_R.append(cociente_norm)
    return cocientes_de_angulos_de_cada_R
    # }}}


def generar_matriz_predicciones(lista_base_network, lista_test_pkl):
    predictions = {i: [] for i in range(len(lista_base_network))}
    for i in range(len(lista_base_network)):
        print(f"{i} en generar_matriz_predicciones")
        test_set, test_tensors_belong_to, base_network = load_weights.load(
            lista_base_network[i], lista_test_pkl[i]
        )
        print(lista_base_network[i])
        predicts = base_network.predict(test_set)
        predictions[i] = [predicts, test_tensors_belong_to, lista_base_network[i]]

    return predictions


# }}}

# {{{ exec
if tipo == "alfa":
    # predicciones_alfa = generar_matriz_predicciones(lista_weights, lista_test_pkl)
    predicciones_alfa = generar_matriz_predicciones(
        lista_weights_nulo, lista_test_pkl_nulo
    )

if tipo == "beta":
    predicciones_beta = generar_matriz_predicciones(lista_weights, lista_test_pkl)

if tipo == "machos":
    predicciones_macho = generar_matriz_predicciones(lista_weights, lista_test_pkl)
# }}}

## }}}

## {{{ generar el csv del tipo cuyas predicciones recién calculé


# {{{ def limpiar_csv(df):
#
def limpiar_csv(df):
    print(df.columns)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    print(df.columns)
    df = df.loc[:, ["tipo", "training_set", "similarities"]]
    print(df.columns)

    return df


# }}}

importlib.reload(metricas)

path_de_la_metrica = "/home/ttdduu/lsd/tesislab/entrenamientos/metricas/gamma/angulos"
if tipo == "alfa":
    # {{{ definir dict (angulos, suma, angulos*suma)
    individuals_alfa = [
        "A_aug",
        "B_aug",
        "19_aug",
        "23_aug",
        "34_aug",
        "HAC1_aug",
        "HAC2_aug",
        "HEC1_aug",
    ]
    dict_medida_calidad = {i: [] for i in trains}  # no son necessarily angulos
    dict_medida_calidad_alfa = para_todas_las_preds(
        predicciones_alfa,  # tiene las predicts (puntos) y el cluster al q pertenece cada uno
        dict_medida_calidad,
        individuals_alfa,
    )
    # }}}

    # {{{ df_alfa.to_csv(
    combined_list_alfa = []

    # Iterate over the values of the dictionary and extend the combined list
    for i in dict_medida_calidad_alfa.values():
        combined_list_alfa.extend(i)
    combined_list_alfa = [i for i in combined_list_alfa if i > 10]

    dict_medida_calidad_alfa = {
        key: value for key, value in dict_medida_calidad_alfa.items() if value
    }  # xq hay empty entries
    data = [
        {"training_set": key, "similarities": item}
        for key, value in dict_medida_calidad_alfa.items()
        for item in value
    ]
    # Create DataFrame
    df_alfa = pd.DataFrame(data)
    df_alfa["tipo"] = "alfa"
    df_alfa.to_csv(f"{path_de_la_metrica}/alfa_nulo.csv")
# }}}

if tipo == "beta":
    # {{{ definir dict (angulos, suma, angulos*suma)
    individuals_beta = ["A", "B", "19", "34", "HAC1", "HAC2", "HEC1"]
    dict_medida_calidad = {i: [] for i in trains}
    dict_medida_calidad_beta = para_todas_las_preds(
        predicciones_beta, dict_medida_calidad, individuals_beta
    )
    # }}}

    # {{{ df_beta.to_csv
    combined_list_beta = []

    # Iterate over the values of the dictionary and extend the combined list
    for i in dict_medida_calidad_beta.values():
        combined_list_beta.extend(i)

    dict_medida_calidad_beta = {
        key: value for key, value in dict_medida_calidad_beta.items() if value
    }  # xq hay empty entries
    data = [
        {"training_set": key, "similarities": item}
        for key, value in dict_medida_calidad_beta.items()
        for item in value
    ]
    # Create DataFrame
    df_beta = pd.DataFrame(data)
    df_beta["tipo"] = "beta"

    df_beta.to_csv(f"{path_de_la_metrica}/beta.csv")
# }}}

if tipo == "machos":
    # {{{ generar dicts_de_angulos, suma y angulos
    individuals_macho = [
        "A",
        "19",
        "AC1",
        "AC2",
        "EC1",
        "B",
        "23",
        "34",
    ]
    dict_medida_calidad_macho = {i: [] for i in trains}
    dict_medida_calidad_macho = para_todas_las_preds(
        predicciones_macho,
        dict_medida_calidad_macho,
        individuals_macho,
    )
    # }}}

    # {{{ to_csv
    combined_list_macho = []

    # Iterate over the values of the dictionary and extend the combined list
    for i in dict_medida_calidad_macho.values():
        combined_list_macho.extend(i)

    # combined_list_macho = [i for i in combined_list_macho if i > 10]

    dict_medida_calidad_macho = {
        key: value for key, value in dict_medida_calidad_macho.items() if value
    }  # xq hay empty entries
    data = [
        {"training_set": key, "similarities": item}
        for key, value in dict_medida_calidad_macho.items()
        for item in value
    ]
    # Create DataFrame
    df_macho = pd.DataFrame(data)
    df_macho["tipo"] = "macho"

    df_macho.to_csv(f"{path_de_la_metrica}/macho.csv")

# }}}
# }}}

## {{{ generar similarities_clean.csv a partir de los tres dfs


# {{{ def df_similarities(alfa, beta, macho):
#
def df_similarities(alfa, beta, macho):
    alfa["tipo"] = "alfa"
    beta["tipo"] = "beta"
    macho["tipo"] = "macho"
    alfa = limpiar_csv(alfa)
    macho = limpiar_csv(macho)
    beta = limpiar_csv(beta)

    df = pd.concat([alfa, beta, macho], ignore_index=True)

    df = df[["tipo", "training_set", "similarities"]]

    df_sorted = df.groupby("tipo", group_keys=False).apply(
        lambda x: x.sort_values(by="similarities")
    )  # cada tipo ordenado de menor a mayor
    return df_sorted


# }}}

# {{{ df.to_csv( --> el de similaridades completo
#
df_alfa = pd.read_csv(f"{path_de_la_metrica}/alfa.csv")
df_alfa = pd.read_csv(f"{path_de_la_metrica}/alfa_nulo.csv")

df_beta = pd.read_csv(f"{path_de_la_metrica}/beta.csv")
df_macho = pd.read_csv(f"{path_de_la_metrica}/macho.csv")

df_similarities(df_alfa, df_beta, df_macho).to_csv(
    f"{path_de_la_metrica}/angulos_con_nulo.csv"
)
# }}}

## }}}


## {{{ plot histogramas de cada set de ángulos
#
# {{{ def limpiar_csv(df):
#
def limpiar_csv(df):
    print(df.columns)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    print(df.columns)
    df = df.loc[:, ["tipo", "training_set", "similarities"]]
    print(df.columns)

    return df


# }}}

df = pd.read_csv(f"{path_de_la_metrica}/angulos.csv")
df = limpiar_csv(df)

import seaborn as sns

plt.figure(figsize=(10, 8))
for tipo in ["alfa", "macho", "beta"]:
    sns.histplot(
        data=df[df["tipo"] == tipo],
        label=tipo,
        x="similarities",
        kde=True,
        bins=30,
    )

plt.legend()
plt.xlabel("gamma")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.show()

## }}}
