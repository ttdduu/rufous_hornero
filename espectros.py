## {{{ import os

from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

## }}}
## {{{ def process_syllable_file(


def process_syllable_file(
    file_path,
):  # --> me da la listita de freccias de la sílaba input
    # Read the txt file and handle 'nan' values
    data = pd.read_csv(f"{file_path}")

    # Extract frequency values and remove NaN values
    frequency_values = data["freq(Hz)"]
    power_values = data["pow(dB/Hz)"]

    return frequency_values, power_values


def filtrar_power(espectro, umbral):
    max_value = espectro["pow(dB/Hz)"].max()
    umbral_rg_din = max_value - umbral  # acá umbral sería el valor del rg dinámico
    # espectro_filtrado = espectro[espectro["pow(dB/Hz)"] >= umbral]
    espectro_filtrado = espectro[espectro["pow(dB/Hz)"] >= umbral_rg_din]
    # espectro_filtrado = espectro

    power = espectro_filtrado["pow(dB/Hz)"]
    frequencies = espectro_filtrado["freq(Hz)"]
    return power, frequencies, espectro_filtrado.index


def relativizar_db(pow):
    # Find the minimum and maximum values in the "Power" column
    min_power = pow.min()
    max_power = pow.max()
    # print(max_power)

    # Scale the "Power" column between 0 and 100
    power = (pow - min_power) / (max_power - min_power)

    return power


def hacer_bins_frecuencia(espectro):
    n_puntos_freq_por_bin = (
        200  # cada punto de freq es cada 5Hz --> cada bin contiene 1000Hz
    )
    bins = (
        [  # recibí la lista de freqs del espectro de la sílaba y la divido en sublistas
            espectro[i : i + n_puntos_freq_por_bin]
            for i in range(0, len(espectro), n_puntos_freq_por_bin)
        ]
    )
    mean_de_cada_bin = [np.mean(i) for i in bins]
    return mean_de_cada_bin


def generar_bd_difs(tipo, individual, power):
    medias = hacer_bins_frecuencia(power)
    medias1, medias2 = medias[: len(medias) // 2], medias[len(medias) // 2 :]
    if len(medias1) != len(medias2):
        medias2 = medias2[1:]
    cocientes = [j - medias1[i] for i, j in enumerate(medias2)]

    return cocientes


## }}}
## {{{ generar base de datos

base_path = "/home/ttdduu/lsd/tesislab/datos/espectros"
tipos = ["macho", "alfa", "beta"]
individuals = ["A", "B", "19", "23", "34", "HEC1", "HAC1", "HAC2"]
difs_espectros = {
    indiv: {tipo: [] for tipo in ["alfa", "beta"]} for indiv in individuals
}
data = []

# filtro_db = [25, 30, 20, 5, 34, 15, 10, 15]
# filtro_db = [66,63, 72,70,71,47, 51,44]
# rg_dinamico = [15] * 3 + [10] * 3 + [15, 15]
filtro_db = [10] * 8
n_del_individuo = {i: {tipo: 0 for tipo in tipos} for i in individuals}

# Iterate through each tipo
for indice_indivs, individual in enumerate(individuals):
    # Iterate through each individual in the tipo
    for tipo in tipos:
        # Construct the path to the individual's directory
        espectros_path = os.path.join(base_path, individual, tipo)
        # Check if the directory exists
        if os.path.exists(espectros_path):
            n_del_individuo[individual][tipo] = (
                n_del_individuo[individual][tipo] + len(os.listdir(espectros_path)) + 1
            )
            for file in os.listdir(espectros_path):
                # Check if the file is a txt file
                if file.endswith(".Table"):
                    # Read the txt file and extract frequency values
                    with open(os.path.join(espectros_path, file), "r") as f:
                        # Extract frequency values from the second column
                        espectro = pd.read_csv(f)
                        power, frequencies, index = filtrar_power(
                            espectro, filtro_db[indice_indivs]
                        )
                        # CURRENTLY acá tengo que hacer lo de las medias

                        filtered_df = espectro[
                            (espectro["freq(Hz)"] >= 1000)
                            & (espectro["freq(Hz)"] <= 7000)
                            & (espectro["pow(dB/Hz)"] > -20)
                        ]
                        power_1k_7k = filtered_df["pow(dB/Hz)"]
                        # print(power_1k_7k)

                        if tipo != "macho":
                            difs_espectros[individual][tipo].extend(
                                generar_bd_difs(tipo, individual, power_1k_7k)
                            )

                        pow_rel = relativizar_db(power)

                        # Extend the data list with the tipo, individual, and frequencies
                        data.extend(
                            [
                                (tipo, individual, frequencies[i], pow_rel[i])
                                for i in index
                            ]
                        )
# pd.DataFrame(n_del_individuo).to_csv("n_de_cada_individuo_espectros.csv")
# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["Tipo", "Individual", "Frequency", "Power"])

# ahora calculo medias +- sd pero limitando las freq entre 2000 y

df_alfa = df[df["Tipo"] == "alfa"].groupby("Individual")
df_beta = df[df["Tipo"] == "beta"].groupby("Individual")

limites_alfa, limites_beta = [3000, 6500], [2000, 4500]

filtered_df_alfa = df[df["Frequency"].between(limites_alfa[0], limites_alfa[1])]
df_alfa_filtered = filtered_df_alfa[filtered_df_alfa["Tipo"] == "alfa"].groupby(
    "Individual"
)
filtered_df_beta = df[df["Frequency"].between(limites_beta[0], limites_beta[1])]
df_beta_filtered = filtered_df_beta[filtered_df_beta["Tipo"] == "beta"].groupby(
    "Individual"
)

mean_freq_alfa = df_alfa_filtered["Frequency"].mean()
sd_freq_alfa = df_alfa_filtered["Frequency"].std()

mean_pow_alfa = df_alfa_filtered["Power"].mean()
sd_pow_alfa = df_alfa_filtered["Power"].std()

mean_freq_beta = df_beta_filtered["Frequency"].mean()
sd_freq_beta = df_beta_filtered["Frequency"].std()

mean_pow_beta = df_beta_filtered["Power"].mean()
sd_pow_beta = df_beta_filtered["Power"].std()

# data = []
# for indiv, values in difs_espectros.items():
#     for tipo, difs in values.items():
#         data.extend(
#             [
#                 {"individuals": indiv, "tipo": tipo, "dB 1er armónico - F0": c}
#                 for c in difs
#             ]
#         )
#
# # Create a DataFrame from the list of dictionaries
# df = pd.DataFrame(data)
# df.to_csv("dif_armonicos-filtrando_dB.csv")
## }}}
## {{{ plot

individuals_grid = [individuals[:4], individuals[4:]]
fig, axes = plt.subplots(4, 2)
for ax in axes.flatten():
    ax.set_xlim(800, 10000)

# plt.tight_layout(h_pad=0.1)
for idx in range(4):
    axes[idx, 0].set_title(f"{individuals[idx]}")
    scatter_alfa_1 = axes[idx, 0].scatter(
        df["Frequency"]
        .loc[df["Individual"] == f"{individuals[idx]}"]
        .loc[df["Tipo"] == "alfa"],
        df["Power"]
        .loc[df["Individual"] == f"{individuals[idx]}"]
        .loc[df["Tipo"] == "alfa"],
        s=5,
        alpha=0.2,
        color="red",
    )
    scatter_beta_1 = axes[idx, 0].scatter(
        df["Frequency"]
        .loc[df["Individual"] == f"{individuals[idx]}"]
        .loc[df["Tipo"] == "beta"],
        df["Power"]
        .loc[df["Individual"] == f"{individuals[idx]}"]
        .loc[df["Tipo"] == "beta"],
        s=5,
        alpha=0.3,
        color="tab:green",
    )
    axes[idx, 0].set_xticks([])
    axes[idx, 0].set_yticks([])

    axes[idx, 0].errorbar(
        mean_freq_alfa[individuals[idx]],
        mean_pow_alfa[individuals[idx]],
        xerr=sd_freq_alfa[individuals[idx]],
        fmt="^",
        markersize=8,
        color="black",
        label=f"alfa: {int(mean_freq_alfa[individuals_grid[0][idx]])}Hz \u00B1 {int(sd_freq_alfa[individuals_grid[0][idx]])}Hz",
    )
    axes[idx, 0].errorbar(
        mean_freq_beta[individuals[idx]],
        mean_pow_beta[individuals[idx]],
        xerr=sd_freq_beta[individuals[idx]],
        fmt="s",
        markersize=8,
        color="black",
        label=f"beta: {int(mean_freq_beta[individuals[idx+4]])}Hz \u00B1 {int(sd_freq_beta[individuals_grid[0][idx]])}Hz",
    )
    axes[idx, 0].legend()

    axes[idx, 1].set_title(f"{individuals[idx+4]}")

    scatter_alfa_2 = axes[idx, 1].scatter(
        df["Frequency"]
        .loc[df["Individual"] == f"{individuals[idx+4]}"]
        .loc[df["Tipo"] == "alfa"],
        df["Power"]
        .loc[df["Individual"] == f"{individuals[idx+4]}"]
        .loc[df["Tipo"] == "alfa"],
        s=5,
        alpha=0.2,
        color="red",
    )
    scatter_beta_2 = axes[idx, 1].scatter(
        df["Frequency"]
        .loc[df["Individual"] == f"{individuals[idx+4]}"]
        .loc[df["Tipo"] == "beta"],
        df["Power"]
        .loc[df["Individual"] == f"{individuals[idx+4]}"]
        .loc[df["Tipo"] == "beta"],
        s=5,
        alpha=0.3,
        color="tab:green",
    )
    axes[idx, 1].set_xticks([])
    axes[idx, 1].set_yticks([])

    axes[idx, 1].errorbar(
        mean_freq_alfa[individuals[idx + 4]],
        mean_pow_alfa[individuals[idx + 4]],
        xerr=sd_freq_alfa[individuals[idx + 4]],
        fmt="^",
        markersize=8,
        color="black",
        label=f"alfa: {int(mean_freq_alfa[individuals_grid[1][idx]])}Hz \u00B1 {int(sd_freq_alfa[individuals_grid[1][idx]])}Hz",
    )

    axes[idx, 1].errorbar(
        mean_freq_beta[individuals[idx + 4]],
        mean_pow_beta[individuals[idx + 4]],
        xerr=sd_freq_beta[individuals[idx + 4]],
        fmt="s",
        markersize=8,
        color="black",
        label=f"beta: {int(mean_freq_beta[individuals_grid[1][idx]])}Hz \u00B1 {int(sd_freq_beta[individuals_grid[1][idx]])}Hz",
    )
    axes[idx, 1].legend()

axes[3, 0].set_xticks(range(2000, 10000, 1000))
axes[3, 1].set_xticks(range(2000, 10000, 1000))
axes[0, 0].set_yticks(range(0, 2))
# Set common labels for all subplots
fig.text(0.5, 0.04, "frecuencia (Hz)", ha="center", va="center")
fig.text(
    0.06,
    0.5,
    "dB/Hz relativizado al máximo de cada sílaba",
    ha="center",
    va="center",
    rotation="vertical",
)
fig.legend(
    [scatter_alfa_1, scatter_beta_1], ["alfa", "beta"], loc="upper right", markerscale=4
)
fig.suptitle("Espectros de sílabas individuales alfa y beta por nido")
## }}}
