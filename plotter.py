# {{{ imports

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import metricas
from sklearn.cluster import KMeans
import load_weights

try:
    from spherecluster import SphericalKMeans
except ImportError:
    pass

# }}}
# {{{ plot silaba


def plot_silaba(file_name):
    print(file_name)
    array = np.genfromtxt(file_name, delimiter=",")
    tiempo, frecuencia = array[:, 0], array[:, 1]
    return tiempo, frecuencia


# }}}
# {{{ def plot_pitch(full_filename):


def plot_pitch(full_filename, ax=None, save=False):
    tiempo = plot_silaba(full_filename)[0]
    frecuencia = plot_silaba(full_filename)[1]
    plt.xlim(0, 0.1295)
    plt.ylim(850, 4000)
    if ax:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(tiempo, frecuencia, color="blue")
    else:
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.plot(tiempo, frecuencia, color="blue")

    if save:
        plt.savefig(f"{full_filename}.png", dpi=50)
        plt.close()


# }}}
# {{{ def p(full_filename):


def p(full_filename):
    return plot_pitch(full_filename)


def pdir(dir, ax=None):  # superponer todas las sílabas de un dir
    file_list = [file for file in os.listdir(dir) if file.endswith(".txt")]
    for filename in file_list:
        full_filepath = os.path.join(dir, filename)
        if ax:
            plot_pitch(full_filepath, ax)
        else:
            plot_pitch(full_filepath)


# }}}
# {{{ def plot2(pitchpaths, pngpaths, suptitle):


def plot2(pitchpaths, pngpaths, suptitle):
    fig, axes = plt.subplots(2, len(pitchpaths), figsize=(8, len(pitchpaths)))
    fig.suptitle(f"{suptitle}")

    if len(pitchpaths) > 1:
        for index, pitchpath in enumerate(pitchpaths):
            img1_path = f"{pngpaths[index]}"
            pitch_path = f"{pitchpath}"
            tiempo = plot_silaba(pitch_path)[0]
            frecuencia = plot_silaba(pitch_path)[1]
            axes[1, index].plot(tiempo, frecuencia)
            axes[1, 0].set_xlabel("tiempo (s)", fontsize=9)
            axes[1, 0].set_ylabel("frecuencia (Hz)", fontsize=9)
            axes[1, index].xaxis.tick_top()
            axes[1, index].grid()
            axes[1, index].tick_params(labelsize=6)
            # axes[0, index].set_title(f'nido {rootname[:rootname.index("-")]}')
            img1 = plt.imread(img1_path)
            axes[0, index].imshow(img1)
            axes[0, index].set_xticks([])
            axes[0, index].set_yticks([])
            plt.show()

    if len(pitchpaths) == 1:
        img1_path = f"{pngpaths[0]}"
        pitch_path = f"{pitchpaths[0]}"
        tiempo = plot_silaba(pitch_path)[0]
        frecuencia = plot_silaba(pitch_path)[1]
        axes[1].plot(tiempo, frecuencia)
        axes[1].set_xlabel("tiempo (s)", fontsize=9)
        axes[1].set_ylabel("frecuencia (Hz)", fontsize=9)
        axes[1].xaxis.tick_top()
        axes[1].grid()
        axes[1].tick_params(labelsize=6)
        # axes[0, index].set_title(f'nido {rootname[:rootname.index("-")]}')
        img1 = plt.imread(img1_path)
        axes[0].imshow(img1)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.show()


# }}}
# {{{ def plot3(sono_paths, sono_pitch_paths, suptitle):


def plot3(sonogs, sonogs_pitch, pitches, suptitle):
    fig, axes = plt.subplots(3, len(pitches), figsize=(8, len(pitches)))
    fig.suptitle(f"{suptitle}")
    ax_titles = [
        "ruido en F0",
        "énfasis en F1 e\n interferencia del macho en un dueto",
        "interferencia del macho en un dueto",
    ]

    for index, pitch_path in enumerate(pitches):
        tiempo = plot_silaba(pitch_path)[0]
        frecuencia = plot_silaba(pitch_path)[1]

        # sonograma solo

        axes[0, index].set_title(ax_titles[index])
        sonog_path = f"{sonogs[index]}"
        sonog = plt.imread(sonog_path)
        axes[0, index].imshow(sonog)
        axes[0, index].set_xticks([])
        axes[0, index].set_yticks([])

        # sonograma con praat con el pitch feo
        axes[1, index].set_title("extracción de pitch en praat")
        sonog_pitch_path = f"{sonogs_pitch[index]}"
        sonog_pitch = plt.imread(sonog_pitch_path)
        axes[1, index].imshow(sonog_pitch)
        axes[1, index].set_xticks([])
        axes[1, index].set_yticks([])

        axes[2, index].set_title("post-procesamiento")
        axes[2, index].plot(tiempo, frecuencia)
        axes[2, 0].set_xlabel("tiempo (s)", fontsize=9)
        axes[2, 0].set_ylabel("frecuencia (Hz)", fontsize=9)
        axes[2, index].xaxis.tick_bottom()
        axes[2, index].grid()
        axes[2, index].tick_params(labelsize=6)
        # axes[0, index].set_title(f'nido {rootname[:rootname.index("-")]}')
        plt.show()


# }}}
# {{{ def plot_imgs_3x2(fila1, fila2, fila3, suptitle):


def plot_imgs_3x2(fila1, fila2, fila3, suptitle="", ax_titles=[]):
    fig, axes = plt.subplots(3, 2, figsize=(8, 2))
    fig.suptitle(f"{suptitle}")

    for columna, foto in enumerate(fila1):
        axes[0, columna].set_title(ax_titles[columna])
        sonog1 = plt.imread(fila1[columna])
        axes[0, columna].imshow(sonog1)
        axes[0, columna].set_xticks([])
        axes[0, columna].set_yticks([])

        sonog2 = plt.imread(fila2[columna])
        axes[1, columna].imshow(sonog2)
        axes[1, columna].set_xticks([])
        axes[1, columna].set_yticks([])

        sonog3 = plt.imread(fila3[columna])
        axes[2, columna].imshow(sonog3)
        axes[2, columna].set_xticks([])
        axes[2, columna].set_yticks([])

        plt.show()


# }}}
# {{{ def plot_3_imgs(imgs, suptitle):


def plot_3_imgs(imgs, suptitle):
    fig, axes = plt.subplots(2)
    fig.suptitle(f"{suptitle}")
    ax_titles = [
        "Sonograma en el rango 1500Hz-6500Hz",
        "F1:Sonograma en el rango 2000Hz-6500Hz",
        "F0: Sonograma en el rango 1000Hz-3250Hz",
    ]

    for fila, foto in enumerate(imgs):
        axes[fila].set_title(ax_titles[fila])
        sonog = plt.imread(foto)
        axes[fila].imshow(sonog)
        axes[fila].set_xticks([])
        axes[fila].set_yticks([])

        plt.show()


# }}}
# {{{ def plot_2x2_imgs
def plot_imgs_2x2(
    fila1, fila2, eje_x_col1, eje_x_col2, eje_y, suptitle="", ax_titles=[]
):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f"{suptitle}")

    for columna, foto in enumerate(fila1):
        axes[0, columna].imshow(plt.imread(fila1[columna]))
        axes[1, columna].imshow(
            plt.imread(fila2[columna]),
            # extent=[
            # 0,
            # plt.imread(fila2[columna]).shape[0],
            # 0,
            # plt.imread(fila2[columna]).shape[1],
            # ],
        )
        axes[0, columna].set_yticks([])
        axes[1, columna].set_yticks([])
        axes[0, columna].set_xticks([])
        axes[1, columna].set_xticks([])

    # Set x-axis ticks and labels for the second row of images
    axes[1, 0].set_xticks(
        np.arange(
            plt.imread(fila2[0]).shape[0],
            step=plt.imread(fila2[columna]).shape[0] / len(eje_x_col1),
        )
    )
    axes[1, 1].set_xticks(
        np.arange(
            plt.imread(fila2[1]).shape[0],
            step=plt.imread(fila2[columna]).shape[0] / len(eje_x_col2),
        )
    )

    axes[1, 0].set_yticks(
        np.arange(
            plt.imread(fila2[0]).shape[1],
            step=plt.imread(fila2[0]).shape[1] / len(eje_y),
        )
    )

    axes[1, 0].set_yticklabels(eje_y)
    axes[1, 0].set_xticklabels(eje_x_col1)
    axes[1, 1].set_xticklabels(eje_x_col1)

    plt.show()


# }}}
# {{{ def plot_superpuestos(lista_dir_pitches,lista_pngs):


def plot_superpuestos(lista_dirs_pitches, lista_pngs):
    fig, axes = plt.subplots(4, 4)
    dirs_arriba, dirs_abajo = lista_dirs_pitches[:4], lista_dirs_pitches[4:]
    pngs_arriba, pngs_abajo = lista_pngs[:4], lista_pngs[4:]

    for index, direc in enumerate(dirs_arriba):
        img1_path = f"{pngs_arriba[index]}"
        indiv_name = direc.split("/")[-1]
        for filename in os.listdir(direc):
            tiempo, frecuencia = plot_silaba(os.path.join(direc, filename))
            axes[1, index].plot(tiempo, frecuencia, color="blue")
            axes[1, index].set_xlim(0, 0.13)  # xaq el vector esté escalado estándar
            axes[3, index].set_ylim(850, 4500)  # xaq el vector esté escalado estándar
            axes[1, index].set_xticks(
                []
            )  # una vez que está estándar, no necesito los ticks
            axes[1, index].set_yticks([])
        img1 = plt.imread(img1_path)
        axes[0, index].imshow(img1, aspect="auto")
        axes[0, index].set_xticks([])
        axes[0, index].set_yticks([])
        axes[0, index].set_title(indiv_name)
        plt.show()

    for index, direc in enumerate(dirs_abajo):
        img1_path = f"{pngs_abajo[index]}"
        indiv_name = direc.split("/")[-1]
        for filename in os.listdir(direc):
            tiempo, frecuencia = plot_silaba(os.path.join(direc, filename))
            axes[3, index].plot(tiempo, frecuencia, color="blue")
            axes[3, index].set_xlim(0, 0.13)  # xaq el vector esté escalado estándar
            axes[3, index].set_ylim(850, 4500)  # xaq el vector esté escalado estándar
            axes[3, index].set_xticks(
                []
            )  # una vez que está estándar, no necesito los ticks
            axes[1, index].set_yticks([])
        img1 = plt.imread(img1_path)
        axes[2, index].imshow(img1, aspect="auto")
        axes[2, index].set_xticks([])
        axes[2, index].set_yticks([])
        axes[2, index].set_title(indiv_name)
        axes[3, index].set_xticks([])
        axes[3, index].set_yticks([])
        plt.show()
    axes[3, 0].set_xticks(np.linspace(0, 0.13, 5))
    axes[3, 0].set_ylabel("Frecuencia (Hz)")
    axes[3, 0].set_yticks(np.linspace(850, 4500, 5))
    axes[3, 0].set_xlabel("Tiempo (s)")
    plt.tight_layout(h_pad=0.5)


# }}}
# {{{ def plot_pkl(fig_name):


def plot_pkl(fig_name):
    with open(fig_name, "rb") as file:
        loaded_fig = pickle.load(file)
    plt.show()


# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/31/scatter-margen_18.pkl" # suma de las distancias entre los centros de masa: 38.6702224323324
#
# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/30/scatter.pkl" # suma de las distancias entre los centros de masa: 36.71648693646598
#
# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/1-primeros/24-unseen-3_epochs/scatter.pkl"
# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/28/scatter-nulo.pkl" # suma de las distancias entre los centros de masa: 10.88459888476795

# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/25/scatter-70_epochs_error_e-08-nice-10_epochs_mas.pkl"

# fig_name="/home/ttdduu/lsd/tesis/datos/entrenamientos/24/scatter-70_epochs_error_e-08.pkl"

# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/34/scatter-margen_18.pkl"

# fig_name = "/home/ttdduu/lsd/tesis/datos/entrenamientos/40/scatter-primero_con_aug-98_epochs.pkl"

# plot_pkl(fig_name)

# }}}
# {{{ def funcs para plotear scatter y resultados de clustereo


def label_colors(test_tensors_belong_to):
    # test_tensors_belong_to = [i[: i.find("_")] for i in test_tensors_belong_to]
    class_labels = set(test_tensors_belong_to)
    colormap = plt.cm.get_cmap("Set1")  # You cpredicts[:, 2]an choose a colormap
    class_to_color = {label: colormap(i) for i, label in enumerate(class_labels)}
    colors = [class_to_color[label] for label in test_tensors_belong_to]
    return class_labels, colormap, colors, class_to_color


def centers_of_mass_colors(
    class_labels, test_tensors_belong_to, predicts, class_to_color
):
    centers_of_mass = np.empty((len(class_labels), 3))

    colors_centers_of_mass_list = []
    for labelnumber, label in enumerate(class_labels):  # centros de masa
        indexes = [
            index
            for index, label in enumerate(f"{label}" == test_tensors_belong_to)
            if label
        ]
        class_predictions = predicts[indexes]
        centers_of_mass[labelnumber] = metricas.center_of_mass(class_predictions)
        colors_centers_of_mass_list.append(
            class_to_color[label]
        )  # xaq tengan el mismo color que el cluster
    return centers_of_mass, colors_centers_of_mass_list


def prediction_plots(
    test_set,  # por default el test set de una sesión interactiva actual
    test_tensors_belong_to,  # same
    base_network,
    path_del_entrenamiento=None,  # same
    descripcion=None,
    max_tests_por_indiv=None,
    ax=None,
    solo_centroides=False,
    legend=False,
    return_centroids=False,
):
    predicts = base_network.predict(test_set)
    norms = np.linalg.norm(predicts, axis=1, keepdims=True)
    predicts = predicts / norms
    x_components = predicts[:, 0]
    y_components = predicts[:, 1]
    z_components = predicts[:, 2]

    class_labels, colormap, colors, class_to_color = label_colors(
        test_tensors_belong_to
    )
    centers_of_mass, colors_centers_of_mass = centers_of_mass_colors(
        class_labels, test_tensors_belong_to, predicts, class_to_color
    )

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    else:
        pass
    if not solo_centroides:
        # scatter puntos
        ax.scatter(x_components, y_components, z_components, c=colors, s=45)
    else:
        pass
    # scatter centros de masa
    ax.scatter(
        centers_of_mass[:, 0],
        centers_of_mass[:, 1],
        centers_of_mass[:, 2],
        c=colors_centers_of_mass,
        s=400,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for label, color in class_to_color.items():
        if "_" in label:
            ax.scatter([], [], [], c=[color], label=label[: label.find("_")])
        else:
            ax.scatter([], [], [], c=[color], label=label)

    #    if legend:
    #        ax.legend(
    #            loc="lower center", fontsize="small", ncol=2, bbox_to_anchor=(0.5, -0.4)
    #        )
    #    else:
    #        ax.legend(
    #            loc="lower center", fontsize="small", ncol=2, bbox_to_anchor=(0.5, -0.4)
    #        )
    ax.legend(loc="lower center", fontsize="small", ncol=2, bbox_to_anchor=(0.5, -0.1))

    # ax.set_title("Clusters de la red")
    if return_centroids:
        return ax, centers_of_mass
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return ax


def plot_k(
    predicts,
    spherical=True,
    ax=None,
    solo_centroides=False,
    return_centroids=False,
    k=8,
):
    print(f"en plot_k: k = {k}")
    if spherical:
        kmeans = SphericalKMeans(n_clusters=k)
    else:
        kmeans = KMeans(n_clusters=k)
    kmeans.fit(predicts)

    # Get the cluster labels
    labels = kmeans.labels_

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        pass

    colormap = plt.cm.get_cmap("Set1")  # You cpredicts[:, 2]an choose a colormap
    # Scatter plot each cluster with different color
    centroids = []
    for i in range(k):  # Assuming you have 8 clusters
        centroid = metricas.center_of_mass(predicts[labels == i])
        centroids.append(centroid)
        xc, yc, zc = [j for j in centroid]
        if not solo_centroides:
            ax.scatter(
                predicts[labels == i, 0],
                predicts[labels == i, 1],
                predicts[labels == i, 2],
                label=f"Cluster {i}",
                color=colormap(i),
                s=45,
            )
        else:
            pass
        ax.scatter(
            xc,
            yc,
            zc,
            # color=colormap(i),
            color="black",
            s=400,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("KMeans Clustering")
    # ax.legend()

    if return_centroids:
        return ax, centroids
    else:
        return ax


def comparar_red_kmeans(
    test_set,  # por default el test set de una sesión interactiva actual
    test_tensors_belong_to,  # same
    base_network,
    axtitles,
    spherical=True,
    completo=True,
    k=8,
):
    predicts = base_network.predict(test_set)

    if completo:
        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)

        ax1.set_title(axtitles[0])
        ax2.set_title(axtitles[1])
        ax3.set_title(axtitles[2])

        prediction_plots(
            test_set=test_set,
            test_tensors_belong_to=test_tensors_belong_to,
            base_network=base_network,
            ax=ax1,
            solo_centroides=False,
        )

        plot_k(predicts, spherical=spherical, ax=ax2, solo_centroides=False, k=k)
        centroides_k = plot_k(
            predicts,
            spherical=spherical,
            ax=ax3,
            solo_centroides=True,
            return_centroids=True,
            k=k,
        )[1]
        print(f"bajo plot_k: k = {k}")

        centroides_red = prediction_plots(
            test_set=test_set,
            test_tensors_belong_to=test_tensors_belong_to,
            base_network=base_network,
            ax=ax3,
            solo_centroides=True,
            legend=False,
            return_centroids=True,
        )[1]

    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(projection="3d")
        ax1.set_title(axtitles[0])
        # plot_k(predicts, spherical=spherical, ax=ax1, solo_centroides=True)
        centroides_k = plot_k(
            predicts,
            spherical=spherical,
            ax=ax1,
            solo_centroides=True,
            return_centroids=True,
            k=k,
        )[1]
        centroides_red = prediction_plots(
            test_set=test_set,
            test_tensors_belong_to=test_tensors_belong_to,
            base_network=base_network,
            ax=ax1,
            solo_centroides=True,
            legend=False,
            return_centroids=True,
        )[1]

    return centroides_k, centroides_red


def plotear_silhouettes(
    lista_base_network,
    lista_test_set,
    spherical=True,
    suptitle=None,
    color="tab:blue",
    ax=None,
    label=None,
    puntos_reales=False,
):  # el arg es una lista de listas; para cada k una lista de coefs
    lista_silhouette_coefficients, lista_real_score = metricas.scores_silhouette(
        lista_base_network, lista_test_set, spherical=True
    )

    # {{{ acá me saco de encima los sils que son de predicts que for some reason son todos puntos del mismo lugar
    A = len(lista_silhouette_coefficients)
    lista_silhouette_coefficients = [
        sublist for sublist in lista_silhouette_coefficients if sublist != [None]
    ]
    B = len(lista_silhouette_coefficients)
    lista_real_score = [i for i in lista_real_score if i != None]
    # lista_real_score = [i for i in lista_real_score if i > 0.47]
    # print(f"a ver ahora {lista_silhouette_coefficients}")
    # print(f"a ver ahora {lista_real_score}")
    print(f"======================trains excluidos por predicts raros = {A - B}")
    # }}}

    filtrar = False
    if filtrar:
        # {{{ outliers con alfa
        if len(lista_silhouette_coefficients) > 1 and "nulo" not in label:
            # if len(lista_real_score) > 1:
            maximos = [
                i[0] for i in lista_silhouette_coefficients
            ]  # silh en posición de k=8
            indices_excluidos1 = [
                index for index, j in enumerate(maximos) if j > 0.68 or j < 0.55
            ]

            maximos2 = [
                i[5] for i in lista_silhouette_coefficients
            ]  # silh en posición k=7
            indices_excluidos2 = [index for index, j in enumerate(maximos2) if j > 0.81]

            maximos3 = [
                i[7] for i in lista_silhouette_coefficients
            ]  # silh en posición k=9 de cada sublista
            indices_excluidos3 = [index for index, j in enumerate(maximos3) if j > 0.82]

            maximos4 = [
                i[9] for i in lista_silhouette_coefficients
            ]  # silh en posición k=9 de cada sublista
            indices_excluidos4 = [
                index for index, j in enumerate(maximos4) if j > 0.78 or j < 0.66
            ]

            maximos5 = [
                i[8] for i in lista_silhouette_coefficients
            ]  # silh en posición k=9 de cada sublista
            indices_excluidos5 = [index for index, j in enumerate(maximos5) if j < 0.7]

            indices_excluidos = (
                # indices_excluidos1
                # + indices_excluidos2
                indices_excluidos3
                + indices_excluidos4
                + indices_excluidos5
            )
            # indices_excluidos = []
            lista_silhouette_coefficients = [
                j
                for i, j in enumerate(lista_silhouette_coefficients)
                if i not in indices_excluidos
            ]
            print(
                f"=========trains excluidos por filtro: {len(indices_excluidos)}: {[lista_base_network[i] for i in indices_excluidos]}"
            )
        #        # }}}

    # {{{ outliers brutos
    if len(lista_silhouette_coefficients) > 1:
        indices_excluidos = []
        for index, lista in enumerate(lista_silhouette_coefficients):
            if any(x < 0.47 for x in lista):  # para no filtrar nada
                indices_excluidos.append(index)
                print("filtré")
        for index, lista in enumerate(lista_silhouette_coefficients):
            if any(x > 0.71 for x in lista):
                if "nulo" in label:
                    print("LA PUTA QUE TE PARIO POR QUE NO FILTRAS NULO")
                indices_excluidos.append(index)
        #
        #        print(indices_excluidos)
        #
        lista_silhouette_coefficients = [
            j
            for i, j in enumerate(lista_silhouette_coefficients)
            if i not in indices_excluidos
        ]

    print(lista_silhouette_coefficients)
    promedios_score_k = [
        np.mean(np.array(lista_silhouette_coefficients)[:, k]) for k in range(13)
    ]

    sd_score_k = [
        np.std(np.array(lista_silhouette_coefficients)[:, k]) for k in range(13)
    ]
    mediana_score_k = [
        np.median(np.array(lista_silhouette_coefficients)[:, k]) for k in range(13)
    ]

    indices_excluidos = []

    #    if any(
    #        i < -0.2 for index, i in enumerate(lista_real_score)
    #    ):  # REVISAR pongo filtro bajito para no filtrar nada
    #        print("LA PUTA QUE TE PARIO POR QUE NO FILTRAS REAL SCORe")
    #        indices_excluidos.append(index)
    #    lista_real_score = [
    #        j for i, j in enumerate(lista_real_score) if i not in indices_excluidos
    #    ]

    # }}}

    # {{{ plotear silhouettes de K
    if not ax:
        fig, ax = plt.subplots(figsize=(24, 7))
        # ax.set_ylim(0,0.95)
        fig.suptitle(suptitle)
    if len(lista_silhouette_coefficients) > 1:
        for i, lista in enumerate(lista_silhouette_coefficients):
            # estas lineas son el grafico de SSEvsK
            if "nulo" in label:
                x = [x + np.random.uniform(0, 0.2) for x in range(2, 15)]
            else:
                x = [x + np.random.uniform(-0.2, 0) for x in range(2, 15)]

            if i == 0:
                ax.plot(
                    x, lista, color=color, label=label, marker="o", linestyle="None"
                )
                ax.set_ylim(-0.8, 0.95)
            else:
                ax.plot(x, lista, color=color, marker="o", linestyle="None")
                ax.set_ylim(-0.8, 0.95)

        if "nulo" not in label:
            ax.errorbar(
                [i - 0.1 for i in range(2, 15)],
                promedios_score_k,
                yerr=sd_score_k,
                color="black",
                label="media \u00B1 sd,modelo real",
                ls="none",
                linewidth=2,
                marker="s",
                ms=8,
            )
        else:
            ax.errorbar(
                [i + 0.1 for i in range(2, 15)],
                promedios_score_k,
                yerr=sd_score_k,
                color="black",
                ls="none",
                linewidth=2,
                label="media \u00B1 sd, modelo nulo",
                marker="^",
                ms=8,
            )
        ax.set_xticks(range(2, 15))
        ax.set_xlabel("Número de clusters")
        ax.set_ylabel("Promedio coeficientes de Silhouette")

    else:
        print(f"en line 757: {lista_silhouette_coefficients}")
        ax.plot(
            range(2, 15),
            lista_silhouette_coefficients,
            color=color,
            marker="o",
            linestyle="None",
        )
        ax.set_xticks(range(2, 15))
        ax.set_ylim(-0.8, 0.95)
        ax.set_xlabel("Número de clusters")
        ax.set_ylabel("Promedio coeficientes de Silhouette")
    # }}}

    # {{{ plotear puntos reales (de lista_real_score)

    if puntos_reales:
        # lista_real_score = [
        # j for i, j in enumerate(lista_real_score) if i > 0.4
        # ]  # not necessary

        """puntos reales"""
        if label:
            if "nulo" in label:
                for i, j in enumerate(lista_real_score):
                    if i == 0:
                        ax.scatter(
                            8 + np.random.uniform(0, 0.2),
                            j,
                            s=40,
                            c="magenta",
                            label="modelo nulo, clusters reales",
                        )
                        ax.set_ylim(-0.8, 0.95)
                        ax.errorbar(
                            8 + 0.1,
                            np.mean(lista_real_score),
                            yerr=np.std(lista_real_score),
                            color="black",
                            label="media \u00B1 sd, modelo real, clusters reales",
                            ls="none",
                            linewidth=2,
                            marker="P",
                            ms=8,
                        )
                    else:
                        ax.scatter(8 + np.random.uniform(0, 0.2), j, s=40, c="magenta")
                        ax.set_ylim(-0.8, 0.95)

            else:
                for i, j in enumerate(lista_real_score):
                    if i == 0:
                        ax.scatter(
                            8, j, s=40, c="green", label="modelo real, clusters reales"
                        )
                        ax.errorbar(
                            8 + 0.1,
                            np.mean(lista_real_score),
                            yerr=np.std(lista_real_score),
                            color="black",
                            label="media \u00B1 sd, modelo nulo, clusters reales",
                            ls="none",
                            linewidth=2,
                            marker="X",
                            ms=8,
                        )
                    else:
                        ax.scatter(8 + np.random.uniform(0, 0.2), j, s=40, c="green")
    # }}}

    ax.legend()
    return (
        ax,
        lista_silhouette_coefficients,
        lista_real_score,
        promedios_score_k,
        sd_score_k,
    )


def plot_silhouette_nulo_vs_real(
    lista_base_network_posta,
    lista_base_network_nulo,
    lista_test_set_posta,
    lista_test_set_nulo,
    suptitle=None,
    puntos_reales=False,
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    (
        axis,
        lista_silhouette_coefficients_real,  # lista de listas, c/u tiene los sil xa cada K de un train
        lista_real_score_real,  # lista con el score de cada train con clusters reales
        promedios_score_k,
        sd_score_k,
    ) = plotear_silhouettes(
        lista_base_network_posta,
        lista_test_set_posta,
        color="tab:blue",
        ax=ax,
        label="modelo real, clusters de KMeans",
        puntos_reales=puntos_reales,
    )
    (
        axis,
        lista_silhouette_coefficients_nulo,
        lista_real_score_nulo,
        promedios_score_k_nulo,
        sd_score_k_nulo,
    ) = plotear_silhouettes(
        lista_base_network_nulo,
        lista_test_set_nulo,
        color="red",
        ax=ax,
        label="modelo nulo, clusters de KMeans",
        puntos_reales=puntos_reales,
    )
    return (
        lista_silhouette_coefficients_real,
        lista_real_score_real,
        lista_silhouette_coefficients_nulo,
        lista_real_score_nulo,
        promedios_score_k,
        sd_score_k,
        promedios_score_k_nulo,
        sd_score_k_nulo,
        fig,
        ax,
    )


def plot_distintos_k(predicts, lista_k, spherical=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")
    axs = [ax1, ax2, ax3, ax4]
    for i, j in enumerate(lista_k):
        plot_k(predicts, spherical=spherical, ax=axs[i], k=lista_k[i])
    fig.suptitle(f"Clusters obtenidos con KMeans para k={lista_k}")


# }}}
# {{{ def histogramas_pitches(individuals = ['A', 'HAC2', '19','23'], num_bins = 100)
def histogramas_pitches(df, tipos, individuals=["A", "HAC2", "19", "23"], num_bins=100):
    # Set the number of bins for the histogram

    # Create a 3x3 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(15, 15), gridspec_kw={"hspace": 0.9})

    # Plot histograms for each individual and tipo
    for i, individual in enumerate(individuals):
        for j, tipo in enumerate(tipos):
            # Filter DataFrame for the current individual and tipo
            filtered_df = df[(df["Individual"] == individual) & (df["Tipo"] == tipo)]

            # Plot histogram
            axes[j, i].hist(filtered_df["Frequency"], bins=num_bins, alpha=0.5)

            # Set labels
            axes[j, 0].set_xlabel("Frequency (Hz)")
            axes[j, 0].set_ylabel("Frequency of each bin")
            axes[j, i].set_title(f"Individuo: {individual}, Tipo: {tipo}")
            axes[j, i].title.set_position([0.5, 1.05])

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


# }}}
