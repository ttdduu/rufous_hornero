import numpy as np
import plotter as p
import load_weights
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

try:
    from spherecluster import SphericalKMeans
except ImportError:
    pass
import warnings

# Suppress UserWarning
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def center_of_mass(
    vectors,
):
    if len(vectors) == 0:
        return None
    return np.mean(vectors, axis=0)


def centros_de_masa(class_labels, test_tensors_belong_to, predicts):
    centers_of_mass = np.empty((len(class_labels), 3))

    for labelnumber, label in enumerate(class_labels):  # centros de masa
        indexes = [
            index
            for index, label in enumerate(f"{label}" == test_tensors_belong_to)
            if label
        ]
        class_predictions = predicts[indexes]
        centers_of_mass[labelnumber] = center_of_mass(class_predictions)
        # print(labelnumber)  # bien
        # print(class_predictions)  # bien
    return centers_of_mass


def sum_distances(matrix):
    num_rows = matrix.shape[0]
    return np.sum(
        np.linalg.norm(matrix[np.newaxis, :, :] - matrix[:, np.newaxis, :], axis=-1)
    )


def kmeans_codo(predicts):
    sse = []  # acá vamos a guardar el puntaje de la función objetivo

    for k in range(1, 15):
        print(k)
        kkmeans = KMeans(n_clusters=k)
        kkmeans.fit(predicts)
        sse.append(kkmeans.inertia_)

    #  REVISAR método del codo: no entendí: k=8 o k=6?

    fig, ax = plt.subplots(figsize=(20, 7))

    # esta dos lineas las agrego para que se vea la elección de KneeLocator para el codo en este gráfico
    ax.scatter(
        8, sse[7], color="red", s=200
    )  # agregamos un punto rojo al plot de tamaño s=200 en el lugar donde se encuentra el codo
    ax.text(
        7.5, sse[7] - 1000, s="codo"
    )  # agregamos un texto abajo para indicar qué representa el punto

    # estas lineas son el grafico de SSEvsK
    ax.scatter(range(1, 15), sse)
    ax.set_xlabel("Número de clusters")
    ax.set_ylabel("SSE")

    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")

    return kl.elbow, fig


def silhouette(base_network, test_set, spherical=True, test_tensors_belong_to=[]):
    predicts = base_network.predict(test_set)
    primera_predict, segunda_predict = predicts[0], predicts[1]
    print(primera_predict, segunda_predict)

    if all(primera_predict[i] == segunda_predict[i] for i in range(3)):
        silhouette_coefficients = [None]
        real_score = None
        return silhouette_coefficients, real_score
    else:
        # Creamos una lista para guardar de los coeficientes de silhouette para cada valor de k
        silhouette_coefficients = []

        # Se necesita tener al menos 2 clusters y a los sumo N-1 (con N el numero de muestras) para obtener coeficientes de Silohuette
        for k in range(2, 15):
            kkkmeans = SphericalKMeans(n_clusters=k)
            kkkmeans.fit(predicts)
            score = silhouette_score(predicts, kkkmeans.labels_, metric="cosine")
            silhouette_coefficients.append(score)

        if test_tensors_belong_to.any():
            real_score = silhouette_score(
                predicts, test_tensors_belong_to, metric="cosine"
            )
            print(f"el real score en silhouette: {real_score}")

            # real_score = silhouette_score(predicts, test_tensors_belong_to, metric="cosine")
            return silhouette_coefficients, real_score
        else:
            return silhouette_coefficients


def scores_silhouette(lista_base_network, lista_test_set, spherical=True):
    lista_lista_coefs = []
    lista_real_score = []
    for i, net in enumerate(lista_base_network):
        print(net)
        test_set, test_tensors_belong_to, base_network = load_weights.load(
            net, lista_test_set[i]
        )
        K, real = silhouette(
            base_network=base_network,
            test_set=test_set,
            spherical=spherical,
            test_tensors_belong_to=test_tensors_belong_to,
        )
        if K:
            lista_real_score.append(real)
            lista_lista_coefs.append(K)
        else:
            print(K)
            print("estuvo mal")
            pass
    return lista_lista_coefs, lista_real_score


import itertools
import math


def distance(point1, point2):
    """Calculate the Euclidean distance between two points in 3D space."""
    return math.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
        + (point2[2] - point1[2]) ** 2
    )


def sum_of_distances(points):
    """Calculate the sum of distances between all unique pairs of points."""
    total_distance = 0
    for pair in itertools.combinations(points, 2):
        total_distance += distance(pair[0], pair[1])
    return total_distance
