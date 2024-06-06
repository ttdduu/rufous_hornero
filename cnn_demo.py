## {{{ imports

import numpy as np
import matplotlib.pyplot as plt

## }}}

## {{{ 1er plot mostrando cómo funcionan los filtros sobre la imagen

fig, axes = plt.subplots(2, 5)
""" FILA 1 """
# {{{ matriz del medio con dos filtros marcados

# {{{ matriz del medio
# Step 1: Create an 8x8 matrix filled with -1
matrix_10x10 = np.full((8, 8), -1)

# Step 2: Set the diagonals to 1
np.fill_diagonal(matrix_10x10, 1)
np.fill_diagonal(np.fliplr(matrix_10x10), 1)

# Step 3: Add padding to create a 10x10 matrix
matrix_10x10 = np.pad(matrix_10x10, pad_width=1, mode="constant", constant_values=-1)
# Plot the matrix
axes[0, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[0, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# {{{ filtros marcados

# el primero
position = (0.5, 0.5)

highlight_rectangle = plt.Rectangle(position, 3, 3, edgecolor="red", lw=6, fill=False)


highlight_rectangle2 = plt.Rectangle(
    tuple(i + 1 for i in position), 1, 1, edgecolor="green", lw=6, fill=False
)

axes[0, 1].add_patch(highlight_rectangle)
axes[0, 1].add_patch(highlight_rectangle2)

# el 2do

position2 = (-0.5, 6.5)
highlight_rectangle = plt.Rectangle(position2, 3, 3, edgecolor="red", lw=6, fill=False)


highlight_rectangle2 = plt.Rectangle(
    tuple(i + 1 for i in position2), 1, 1, edgecolor="cyan", lw=6, fill=False
)

axes[0, 1].add_patch(highlight_rectangle)
axes[0, 1].add_patch(highlight_rectangle2)

# }}}

# }}}

# {{{ filtrito de la izq

filtro = np.array([[1, 1, -1], [-1, 1, -1], [-1, -1, 1]])

axes[0, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[0,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[0, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="red", lw=6, fill=False
)
axes[0, 0].add_patch(highlight_rectangle)
axes[0, 0].axis("off")
# }}}

# {{{ feature map

# Define the size of the output feature map
feature_map_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map = np.zeros(feature_map_size)

# Perform convolution
for i in range(feature_map_size[0]):
    for j in range(feature_map_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map[i, j] = round(convolution_result / 9, 2)

axes[0, 2].imshow(
    feature_map,
    cmap="Reds_r",
    interpolation="nearest",
)
axes[0, 2].set_box_aspect(10 / 10)
axes[0, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map.shape[0]):
    for j in range(feature_map.shape[1]):
        value = feature_map[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle_filtro1 = plt.Rectangle(
    position, 1, 1, edgecolor="green", lw=6, fill=False
)

"""
el parche completo en esa ubicación, que puede ser identificada con su centro, es un
perfect match con el filtro --> a esa ubicación le corresponde un score de 1 en el mapa
de este filtro.
"""
axes[0, 2].add_patch(highlight_rectangle_filtro1)

highlight_rectangle_filtro2 = plt.Rectangle(
    position2, 1, 1, edgecolor="cyan", lw=6, fill=False
)
axes[0, 2].add_patch(highlight_rectangle_filtro2)

# }}}

"""FILA 2"""
# {{{ matriz del medio con dos filtros marcados

# {{{ filtrito de la izq

filtro = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

axes[1, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[1,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[1, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="blue", lw=6, fill=False
)
axes[1, 0].add_patch(highlight_rectangle)
axes[1, 0].axis("off")
# }}}

# {{{ matriz del medio
# Step 1: Create an 8x8 matrix filled with -1
matrix_10x10 = np.full((8, 8), -1)

# Step 2: Set the diagonals to 1
np.fill_diagonal(matrix_10x10, 1)
np.fill_diagonal(np.fliplr(matrix_10x10), 1)

# Step 3: Add padding to create a 10x10 matrix
matrix_10x10 = np.pad(matrix_10x10, pad_width=1, mode="constant", constant_values=-1)
# Plot the matrix
axes[1, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[1, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# {{{ filtros marcados

# el primero
position = (0.5, 0.5)

highlight_rectangle = plt.Rectangle(position, 3, 3, edgecolor="blue", lw=6, fill=False)


highlight_rectangle2 = plt.Rectangle(
    tuple(i + 1 for i in position), 1, 1, edgecolor="magenta", lw=6, fill=False
)

axes[1, 1].add_patch(highlight_rectangle)
axes[1, 1].add_patch(highlight_rectangle2)

# el 2do

position2 = (-0.5, 6.5)
highlight_rectangle = plt.Rectangle(position2, 3, 3, edgecolor="blue", lw=6, fill=False)


highlight_rectangle2 = plt.Rectangle(
    tuple(i + 1 for i in position2), 1, 1, edgecolor="yellow", lw=6, fill=False
)

axes[1, 1].add_patch(highlight_rectangle)
axes[1, 1].add_patch(highlight_rectangle2)

# }}}

# {{{ feature map

# Define the size of the output feature map
feature_map_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map = np.zeros(feature_map_size)

# Perform convolution
for i in range(feature_map_size[0]):
    for j in range(feature_map_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map[i, j] = round(convolution_result / 9, 2)

axes[1, 2].imshow(
    feature_map,
    cmap="Blues_r",
    interpolation="nearest",
)
axes[1, 2].set_box_aspect(10 / 10)
axes[1, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map.shape[0]):
    for j in range(feature_map.shape[1]):
        value = feature_map[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle_filtro1 = plt.Rectangle(
    position, 1, 1, edgecolor="magenta", lw=6, fill=False
)

"""
el parche completo en esa ubicación, que puede ser identificada con su centro, es un
perfect match con el filtro --> a esa ubicación le corresponde un score de 1 en el mapa
de este filtro.
"""
axes[1, 2].add_patch(highlight_rectangle_filtro1)

highlight_rectangle_filtro2 = plt.Rectangle(
    position2, 1, 1, edgecolor="yellow", lw=6, fill=False
)
axes[1, 2].add_patch(highlight_rectangle_filtro2)

# }}}
# }}}


axes[0, 0].set_title("A) Filtros convolucionales")
axes[0, 1].set_title("B) Imagen original")
axes[0, 2].set_title("C) Imagen filtrada")
## }}}
## {{{ 2do plot mostrando cómo se hace el maxpooling y el aplanado

fig, axes = plt.subplots(2, 5)
""" FILA 1 """
# {{{ filtrito de la izq

filtro = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

axes[0, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[0,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[0, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="red", lw=6, fill=False
)
axes[0, 0].add_patch(highlight_rectangle)
axes[0, 0].axis("off")
# }}}

# {{{ matriz del medio sin marcarle los filtros

# {{{ matriz del medio
# Step 1: Create an 8x8 matrix filled with -1
matrix_10x10 = np.full((8, 8), -1)

# Step 2: Set the diagonals to 1
np.fill_diagonal(matrix_10x10, 1)
np.fill_diagonal(np.fliplr(matrix_10x10), 1)

# Step 3: Add padding to create a 10x10 matrix
matrix_10x10 = np.pad(matrix_10x10, pad_width=1, mode="constant", constant_values=-1)
# Plot the matrix
axes[0, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[0, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# }}}

# {{{ feature map sin marcarle los lugarcitos

# Define the size of the output feature map
feature_map1_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map1 = np.zeros(feature_map1_size)

# Perform convolution
for i in range(feature_map1_size[0]):
    for j in range(feature_map1_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map1[i, j] = round(convolution_result / 9, 2)

axes[0, 2].imshow(
    feature_map1,
    cmap="Reds_r",
    interpolation="nearest",
)
axes[0, 2].set_box_aspect(10 / 10)
axes[0, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map1.shape[0]):
    for j in range(feature_map1.shape[1]):
        value = feature_map1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


# }}}

"""FILA 2"""
# {{{ filtrito de la izq

filtro = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

axes[1, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[1,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[1, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="blue", lw=6, fill=False
)
axes[1, 0].add_patch(highlight_rectangle)
axes[1, 0].axis("off")
# }}}

# {{{ matriz del medio sin marcarle los filtros
# Step 1: Create an 8x8 matrix filled with -1
matrix_10x10 = np.full((8, 8), -1)

# Step 2: Set the diagonals to 1
np.fill_diagonal(matrix_10x10, 1)
np.fill_diagonal(np.fliplr(matrix_10x10), 1)

# Step 3: Add padding to create a 10x10 matrix
matrix_10x10 = np.pad(matrix_10x10, pad_width=1, mode="constant", constant_values=-1)
# Plot the matrix
axes[1, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[1, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# {{{ feature map

# Define the size of the output feature map
feature_map2_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map2 = np.zeros(feature_map2_size)

# Perform convolution
for i in range(feature_map2_size[0]):
    for j in range(feature_map2_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map2[i, j] = round(convolution_result / 9, 2)

axes[1, 2].imshow(
    feature_map2,
    cmap="Blues_r",
    interpolation="nearest",
)
axes[1, 2].set_box_aspect(10 / 10)
axes[1, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map2.shape[0]):
    for j in range(feature_map2.shape[1]):
        value = feature_map2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}

"""
maxpooling en última columna
"""
# {{{ arriba

# Define max pooling window size
window_size = 2

# Calculate output size after max pooling
output_size = (
    feature_map1.shape[0] // window_size,
    feature_map1.shape[1] // window_size,
)

# Initialize the pooled matrix
pooled_matrix1 = np.zeros(output_size)

# Perform max pooling
for i in range(0, feature_map1.shape[0], window_size):
    for j in range(0, feature_map1.shape[1], window_size):
        window = feature_map1[i : i + window_size, j : j + window_size]
        pooled_matrix1[i // window_size, j // window_size] = np.max(window)

axes[0, 3].imshow(
    pooled_matrix1,
    cmap="Reds_r",
    interpolation="nearest",
)
# axes[0, 3].set_box_aspect(4 / 10)
axes[0, 3].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 3].set_xticks(np.arange(-0.5, 4, 1), [])
axes[0, 3].set_yticks(np.arange(-0.5, 4, 1), [])
axes[0, 3].set_xticks(np.arange(-0.5, 4, 1), minor=True)
axes[0, 3].set_yticks(np.arange(-0.5, 4, 1), minor=True)
axes[0, 3].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(pooled_matrix1.shape[0]):
    for j in range(pooled_matrix1.shape[1]):
        value = pooled_matrix1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 3].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}
# {{{ abajo
# Define max pooling window size
window_size = 2

# Calculate output size after max pooling
output_size = (
    feature_map2.shape[0] // window_size,
    feature_map2.shape[1] // window_size,
)

# Initialize the pooled matrix
pooled_matrix2 = np.zeros(output_size)

# Perform max pooling
for i in range(0, feature_map2.shape[0], window_size):
    for j in range(0, feature_map2.shape[1], window_size):
        window = feature_map2[i : i + window_size, j : j + window_size]
        pooled_matrix2[i // window_size, j // window_size] = np.max(window)

axes[1, 3].imshow(
    pooled_matrix2,
    cmap="Blues_r",
    interpolation="nearest",
)
# axes[1, 3].set_box_aspect(4 / 10)
axes[1, 3].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 3].set_xticks(np.arange(-0.5, 4, 1), [])
axes[1, 3].set_yticks(np.arange(-0.5, 4, 1), [])
axes[1, 3].set_xticks(np.arange(-0.5, 4, 1), minor=True)
axes[1, 3].set_yticks(np.arange(-0.5, 4, 1), minor=True)
axes[1, 3].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(pooled_matrix2.shape[0]):
    for j in range(pooled_matrix2.shape[1]):
        value = pooled_matrix2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 3].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}
# {{{ agregar la ventanita de maxpooling en el map de arriba y en el pool

position = (3.5, 1.5)
highlight_rectangle = plt.Rectangle(position, 2, 2, edgecolor="lime", lw=4, fill=False)
axes[0, 2].add_patch(highlight_rectangle)
highlight_rectangle = plt.Rectangle(
    (1.5, 0.5), 1, 1, edgecolor="lime", lw=4, fill=False
)
axes[0, 3].add_patch(highlight_rectangle)


axes[0, 0].set_title("A) Filtros convolucionales")
axes[0, 1].set_title("B) Imagen original")
axes[0, 2].set_title("C) Imagen filtrada")
axes[0, 3].set_title("D) Maxpooling (2x2) de la imagen filtrada")


# {{{ aplanado
# {{{ arriba

# Initialize flattened vectors for both matrices
flattened_vector2 = []
# Flatten pooled_matrix2
for i in range(pooled_matrix2.shape[0]):
    for j in range(pooled_matrix2.shape[1]):
        value = pooled_matrix2[i, j]
        # color = "blue" if value < 0.5 else "white"  # Adjust color threshold as needed
        flattened_vector2.append(value)


flattened_vector2 = np.array(flattened_vector2).reshape(16, 1)
axes[0, 4].imshow(
    flattened_vector2,
    cmap="Reds_r",
    interpolation="nearest",
)

highlight_rectangle = plt.Rectangle(
    (-0.5, 4.5), 1, 1, edgecolor="lime", lw=4, fill=False
)
axes[0, 4].add_patch(highlight_rectangle)

for i in range(flattened_vector2.shape[0]):
    for j in range(flattened_vector2.shape[1]):
        value = flattened_vector2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 4].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


box = axes[0, 4].get_position()
axes[0, 4].set_position([box.x0, box.y0, box.width * 1.1, box.height * 1.2])


axes[0, 4].set_xticks([])
axes[0, 4].set_yticks([])

# }}}
# {{{ abajo
# Initialize flattened vectors for both matrices
flattened_vector1 = []

# Flatten pooled_matrix1
for i in range(pooled_matrix1.shape[0]):
    for j in range(pooled_matrix1.shape[1]):
        value = pooled_matrix1[i, j]
        # color = "red" if value < 0.5 else "white"  # Adjust color threshold as needed
        flattened_vector1.append(value)

flattened_vector1 = np.array(flattened_vector1).reshape(16, 1)
axes[1, 4].imshow(
    flattened_vector1,
    cmap="Blues_r",
    interpolation="nearest",
)

for i in range(flattened_vector1.shape[0]):
    for j in range(flattened_vector1.shape[1]):
        value = flattened_vector1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 4].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

box = axes[1, 4].get_position()
axes[1, 4].set_position([box.x0, box.y0, box.width * 1.1, box.height * 1.2])

# }}}

axes[1, 4].set_xticks([])
axes[1, 4].set_yticks([])
# axes[0,4].set_xlim(0, len(flattened_vector1))
# axes[0,4].set_ylim(0, 1)
# axes[0,4].axis('off')
axes[0, 4].set_title("E) Aplanado")
# }}}

## }}}
## }}}
## {{{ O

fig, axes = plt.subplots(2, 5)
""" FILA 1 """
# {{{ filtrito de la izq

filtro = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

axes[0, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[0,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[0, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[0, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[0, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="red", lw=6, fill=False
)
axes[0, 0].add_patch(highlight_rectangle)
axes[0, 0].axis("off")
# }}}

# {{{ matriz del medio
# Step 1: Create an 8x8 matrix filled with -1
matrix_8x8 = np.full((8, 8), -1)
matrix_8x8[1:-1, 1] = 1
matrix_8x8[1:-1, -2] = 1
matrix_8x8[1, 1:-1] = 1
matrix_8x8[-2, 1:-1] = 1

# Add padding to make it a 10x10 matrix
matrix_10x10 = np.full((10, 10), -1)
matrix_10x10[1:-1, 1:-1] = matrix_8x8
# Plot the matrix
axes[0, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[0, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[0, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[0, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[0, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# {{{ feature map sin marcarle los lugarcitos

# Define the size of the output feature map
feature_map1_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map1 = np.zeros(feature_map1_size)

# Perform convolution
for i in range(feature_map1_size[0]):
    for j in range(feature_map1_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map1[i, j] = round(convolution_result / 9, 2)

axes[0, 2].imshow(
    feature_map1,
    cmap="Reds_r",
    interpolation="nearest",
)
axes[0, 2].set_box_aspect(10 / 10)
axes[0, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[0, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[0, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map1.shape[0]):
    for j in range(feature_map1.shape[1]):
        value = feature_map1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


# }}}

"""FILA 2"""
# {{{ filtrito de la izq

filtro = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

axes[1, 0].imshow(filtro, cmap="gray", interpolation="nearest")
# axes[1,0].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), [])
axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].set_yticks(np.arange(-0.5, 3, 1), minor=True)
axes[1, 0].grid(which="minor", color="black", linestyle="-", linewidth=1)
axes[1, 0].set_box_aspect(10 / 3)

for i in range(filtro.shape[0]):
    for j in range(filtro.shape[1]):
        value = filtro[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 0].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


highlight_rectangle = plt.Rectangle(
    (-0.5, -0.5), 3, 3, edgecolor="blue", lw=6, fill=False
)
axes[1, 0].add_patch(highlight_rectangle)
axes[1, 0].axis("off")
# }}}

# {{{ matriz del medio
# Step 1: Create an 8x8 matrix filled with -1
matrix_8x8 = np.full((8, 8), -1)
matrix_8x8[1:-1, 1] = 1
matrix_8x8[1:-1, -2] = 1
matrix_8x8[1, 1:-1] = 1
matrix_8x8[-2, 1:-1] = 1

# Add padding to make it a 10x10 matrix
matrix_10x10 = np.full((10, 10), -1)
matrix_10x10[1:-1, 1:-1] = matrix_8x8
# Plot the matrix
axes[1, 1].imshow(matrix_10x10, cmap="gray", interpolation="nearest")
axes[1, 1].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), [])
axes[1, 1].set_xticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].set_yticks(np.arange(-0.5, 10, 1), minor=True)
axes[1, 1].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(matrix_10x10.shape[0]):
    for j in range(matrix_10x10.shape[1]):
        value = matrix_10x10[i, j]
        color = "white" if value == -1 else "black"
        axes[1, 1].text(j, i, value, ha="center", va="center", color=color, fontsize=12)
# }}}

# {{{ feature map

# Define the size of the output feature map
feature_map2_size = (
    matrix_10x10.shape[0] - filtro.shape[0] + 1,
    matrix_10x10.shape[1] - filtro.shape[1] + 1,
)

# Initialize the feature map
feature_map2 = np.zeros(feature_map2_size)

# Perform convolution
for i in range(feature_map2_size[0]):
    for j in range(feature_map2_size[1]):
        patch = matrix_10x10[i : i + filtro.shape[0], j : j + filtro.shape[1]]
        convolution_result = np.sum(np.multiply(patch, filtro))
        feature_map2[i, j] = round(convolution_result / 9, 2)

axes[1, 2].imshow(
    feature_map2,
    cmap="Blues_r",
    interpolation="nearest",
)
axes[1, 2].set_box_aspect(10 / 10)
axes[1, 2].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), [])
axes[1, 2].set_xticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
axes[1, 2].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(feature_map2.shape[0]):
    for j in range(feature_map2.shape[1]):
        value = feature_map2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 2].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}

"""
maxpooling en última columna
"""
# {{{ arriba

# Define max pooling window size
window_size = 2

# Calculate output size after max pooling
output_size = (
    feature_map1.shape[0] // window_size,
    feature_map1.shape[1] // window_size,
)

# Initialize the pooled matrix
pooled_matrix1 = np.zeros(output_size)

# Perform max pooling
for i in range(0, feature_map1.shape[0], window_size):
    for j in range(0, feature_map1.shape[1], window_size):
        window = feature_map1[i : i + window_size, j : j + window_size]
        pooled_matrix1[i // window_size, j // window_size] = np.max(window)

axes[0, 3].imshow(
    pooled_matrix1,
    cmap="Reds_r",
    interpolation="nearest",
)
# axes[0, 3].set_box_aspect(4 / 10)
axes[0, 3].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[0, 3].set_xticks(np.arange(-0.5, 4, 1), [])
axes[0, 3].set_yticks(np.arange(-0.5, 4, 1), [])
axes[0, 3].set_xticks(np.arange(-0.5, 4, 1), minor=True)
axes[0, 3].set_yticks(np.arange(-0.5, 4, 1), minor=True)
axes[0, 3].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(pooled_matrix1.shape[0]):
    for j in range(pooled_matrix1.shape[1]):
        value = pooled_matrix1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 3].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}
# {{{ abajo
# Define max pooling window size
window_size = 2

# Calculate output size after max pooling
output_size = (
    feature_map2.shape[0] // window_size,
    feature_map2.shape[1] // window_size,
)

# Initialize the pooled matrix
pooled_matrix2 = np.zeros(output_size)

# Perform max pooling
for i in range(0, feature_map2.shape[0], window_size):
    for j in range(0, feature_map2.shape[1], window_size):
        window = feature_map2[i : i + window_size, j : j + window_size]
        pooled_matrix2[i // window_size, j // window_size] = np.max(window)

axes[1, 3].imshow(
    pooled_matrix2,
    cmap="Blues_r",
    interpolation="nearest",
)
# axes[1, 3].set_box_aspect(4 / 10)
axes[1, 3].grid(True, which="both", color="gray", linestyle="-", linewidth=1)
# Configure the gridlines to align with the matrix cells
axes[1, 3].set_xticks(np.arange(-0.5, 4, 1), [])
axes[1, 3].set_yticks(np.arange(-0.5, 4, 1), [])
axes[1, 3].set_xticks(np.arange(-0.5, 4, 1), minor=True)
axes[1, 3].set_yticks(np.arange(-0.5, 4, 1), minor=True)
axes[1, 3].grid(which="minor", color="black", linestyle="-", linewidth=1)

for i in range(pooled_matrix2.shape[0]):
    for j in range(pooled_matrix2.shape[1]):
        value = pooled_matrix2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 3].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

# }}}
# {{{ agregar la ventanita de maxpooling en el map de arriba y en el pool

position = (3.5, 1.5)
highlight_rectangle = plt.Rectangle(position, 2, 2, edgecolor="lime", lw=4, fill=False)
axes[0, 2].add_patch(highlight_rectangle)
highlight_rectangle = plt.Rectangle(
    (1.5, 0.5), 1, 1, edgecolor="lime", lw=4, fill=False
)
axes[0, 3].add_patch(highlight_rectangle)


axes[0, 0].set_title("A) Filtros convolucionales")
axes[0, 1].set_title("B) Imagen original")
axes[0, 2].set_title("C) Imagen filtrada")
axes[0, 3].set_title("D) Maxpooling (2x2) de la imagen filtrada")


# {{{ aplanado
# {{{ arriba

# Initialize flattened vectors for both matrices
flattened_vector2 = []
# Flatten pooled_matrix2
for i in range(pooled_matrix2.shape[0]):
    for j in range(pooled_matrix2.shape[1]):
        value = pooled_matrix2[i, j]
        # color = "blue" if value < 0.5 else "white"  # Adjust color threshold as needed
        flattened_vector2.append(value)


flattened_vector2 = np.array(flattened_vector2).reshape(16, 1)
axes[0, 4].imshow(
    flattened_vector2,
    cmap="Reds_r",
    interpolation="nearest",
)

highlight_rectangle = plt.Rectangle(
    (-0.5, 4.5), 1, 1, edgecolor="lime", lw=4, fill=False
)
axes[0, 4].add_patch(highlight_rectangle)

for i in range(flattened_vector2.shape[0]):
    for j in range(flattened_vector2.shape[1]):
        value = flattened_vector2[i, j]
        color = "white" if value < 0.5 else "black"
        axes[0, 4].text(j, i, value, ha="center", va="center", color=color, fontsize=8)


box = axes[0, 4].get_position()
axes[0, 4].set_position([box.x0, box.y0, box.width * 1.1, box.height * 1.2])


axes[0, 4].set_xticks([])
axes[0, 4].set_yticks([])

# }}}
# {{{ abajo
# Initialize flattened vectors for both matrices
flattened_vector1 = []

# Flatten pooled_matrix1
for i in range(pooled_matrix1.shape[0]):
    for j in range(pooled_matrix1.shape[1]):
        value = pooled_matrix1[i, j]
        # color = "red" if value < 0.5 else "white"  # Adjust color threshold as needed
        flattened_vector1.append(value)

flattened_vector1 = np.array(flattened_vector1).reshape(16, 1)
axes[1, 4].imshow(
    flattened_vector1,
    cmap="Blues_r",
    interpolation="nearest",
)

for i in range(flattened_vector1.shape[0]):
    for j in range(flattened_vector1.shape[1]):
        value = flattened_vector1[i, j]
        color = "white" if value < 0.5 else "black"
        axes[1, 4].text(j, i, value, ha="center", va="center", color=color, fontsize=8)

box = axes[1, 4].get_position()
axes[1, 4].set_position([box.x0, box.y0, box.width * 1.1, box.height * 1.2])

# }}}

axes[1, 4].set_xticks([])
axes[1, 4].set_yticks([])
# axes[0,4].set_xlim(0, len(flattened_vector1))
# axes[0,4].set_ylim(0, 1)
# axes[0,4].axis('off')
axes[0, 4].set_title("E) Aplanado")
# }}}

## }}}
## }}}
