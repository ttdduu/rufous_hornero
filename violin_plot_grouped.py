## {{{ import os

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## }}}
## {{{ def process_syllable_file(


def process_syllable_file(
    file_path,
):  # --> me da la listita de freccias de la sílaba input
    # Read the txt file and handle 'nan' values
    data = np.genfromtxt(
        file_path, delimiter=",", missing_values="nan", filling_values=np.nan
    )

    # Extract frequency values and remove NaN values
    frequency_values = data[:, 1]
    frequency_values = frequency_values[~np.isnan(frequency_values)]

    return frequency_values


## }}}}
## {{{ # Create a DataFrame from the data list para alfa-beta-macho

# Define the base path

base_path = "/home/ttdduu/lsd/tesislab/datos/pitches/violinplot"

# Initialize an empty list to store frequency values for each tipo and individual
data = []

# Define the types of individuals
individuals = ["A", "B", "19", "23", "34", "HEC1", "HAC1", "HAC2"]

n_de_silabas_de_cada_indiv = {"macho": 0, "alfa": 0, "beta": 0}
# Iterate through each tipo
for tipo in ["macho", "alfa", "beta"]:
    # Iterate through each individual in the tipo
    for individual in individuals:
        # Construct the path to the individual's directory
        individual_path = os.path.join(base_path, tipo, individual)
        # Check if the directory exists
        if os.path.exists(individual_path):
            # Iterate through each directory (songs) in the individual's directory
            for songs_dir in os.listdir(individual_path):
                # Construct the path to the songs directory
                songs_path = os.path.join(individual_path, songs_dir)
                # Check if it is a directory
                if os.path.isdir(songs_path):
                    # Iterate through each file in the songs directory
                    n_de_silabas_de_cada_indiv[tipo] = n_de_silabas_de_cada_indiv[
                        tipo
                    ] + len(os.listdir(songs_path))
                    for file in os.listdir(songs_path):
                        # Check if the file is a txt file
                        if file.endswith(".txt"):
                            # Read the txt file and extract frequency values
                            with open(os.path.join(songs_path, file), "r") as f:
                                # Extract frequency values from the second column
                                frequencies = process_syllable_file(f)
                                # Extend the data list with the tipo, individual, and frequencies
                                data.extend(
                                    [(tipo, individual, freq) for freq in frequencies]
                                )

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["Tipo", "Individual", "Frequency"])

mean_by_individual = df[df["Tipo"] == "beta"].groupby("Individual")["Frequency"].mean()
sd_by_individual = df[df["Tipo"] == "beta"].groupby("Individual")["Frequency"].std()

meanb, sdb = (
    df[df["Tipo"] == "beta"].groupby("Individual")["Frequency"].mean(),
    df[df["Tipo"] == "beta"].groupby("Individual")["Frequency"].std(),
)

meana, sda = (
    df[df["Tipo"] == "alfa"].groupby("Individual")["Frequency"].mean(),
    df[df["Tipo"] == "alfa"].groupby("Individual")["Frequency"].std(),
)

meanm, sdm = (
    df[df["Tipo"] == "macho"].groupby("Individual")["Frequency"].mean(),
    df[df["Tipo"] == "macho"].groupby("Individual")["Frequency"].std(),
)

# Update the DataFrame with the calculated mean values
df["Mean_Frequency"] = df["Individual"].map(mean_by_individual)

# Add the mean frequency value to every frequency value of tipo alfa for each individual
df.loc[df["Tipo"] == "alfa", "Frequency"] += df.loc[
    df["Tipo"] == "alfa", "Mean_Frequency"
]
updated_mean_by_individual = (
    df[df["Tipo"] == "alfa"].groupby("Individual")["Frequency"].mean()
)


""" no va
# Apply the filter function to each group
df = df[df["Tipo"] != "macho"]

# Separate rows with Individual 'A' and 'HAC2'
df_A = df[df["Individual"] == "A"]
df_HAC2 = df[df["Individual"] == "HAC2"]
df_HAC1 = df[df["Individual"] == "HAC1"]
Randomly select half of the rows for each group
np.random.seed(42)  for reproducibility
selected_rows_A = np.random.choice(df_A.index, size=len(df_A) // 10, replace=False)
selected_rows_HAC2 = np.random.choice(
    df_HAC2.index, size=len(df_HAC2) // 10, replace=False
)
selected_rows_HAC1 = np.random.choice(
    df_HAC1.index, size=len(df_HAC1) // 10, replace=False
)

Concatenate the selected rows back together
filtered_df = pd.concat(
    [
        df_A.loc[selected_rows_A],
        df_HAC1.loc[selected_rows_HAC1],
        df_HAC2.loc[selected_rows_HAC2],
        df[~df["Individual"].isin(["A", "HAC2", "HAC1"])],
    ]
)


Function to filter rows within 90% quantile range for each Tipo
def filter_quantile(group):
    quantile_10 = group["Frequency"].quantile(0.2)
    quantile_90 = group["Frequency"].quantile(0.95)
    return group[
        (group["Frequency"] >= quantile_10) & (group["Frequency"] <= quantile_90)
    ]


Apply the filter function to each group separated by Individual and Tipo
filtered_df = filtered_df.groupby(["Individual", "Tipo"]).apply(filter_quantile)
filtered_df.to_csv("datos_violinplot_80iq.csv")
"""
len(filtered_df)
## }}}
## {{{ csv de medias+-sd

# df.to_csv(
# "/home/ttdduu/lsd/tesislab/datos/pitches/violinplot/datos_crudos_violinplot.csv"
# )

stats_beta = pd.DataFrame(
    {
        "Individual": meana.index,
        "Media (Hz)": meana.astype(int),
        "SD (Hz)": sda.astype(int),
    }
).reset_index(drop=True)

# stats_beta.to_csv(
# "/home/ttdduu/lsd/tesislab/datos/pitches/violinplot/medias_violin_beta.csv",
# index=False,
# )

## }}}
## {{{ # Create a violin plot using seaborn para alfa-beta-macho
from numpy import mean

# Create a violin plot using seaborn
individual_palette = {"alfa": "red", "beta": "tab:green", "macho": "blue"}

plt.figure(figsize=(12, 8))
ax = sns.violinplot(
    data=df,
    x="Individual",
    y="Frequency",
    hue="Tipo",
    # split=True,
    inner=None,
    palette=individual_palette,
    inner_estimator=None,
)
# for patch in ax.artists:
# r, g, b, a = patch.get_facecolor()
# patch.set_facecolor((r, g, b, .7))  # Adjust the alpha (transparency) as needed

# Customize the color of the points representing the mean
# for line in ax.lines:
# line.set_color('white')  # Set the color of the points representing the mean

for i in range(len(individuals) - 1):
    plt.axvline(x=i + 0.5, color="gray", linestyle="--")

for i, group_name in enumerate(df["Individual"].unique()):
    for j, tipo in enumerate(["macho", "alfa", "beta"]):
        group_data = df[(df["Individual"] == group_name) & (df["Tipo"] == tipo)][
            "Frequency"
        ]
        group_std = np.std(group_data)
        group_mean = np.mean(group_data)
        x_offset = i + (j - 1) * 0.27  # Adjust the multiplier as needed
        (_, caps, _) = ax.errorbar(
            x_offset,
            np.mean(group_data),
            yerr=group_std,
            fmt="none",
            color="white",
            capsize=10,
            capthick=2,
            lw=2,
        )
        ax.scatter(x_offset, group_mean, c="black", s=10, zorder=3)
        for cap in caps:
            cap.set_color("black")


plt.title("Distribución de valores de frecuencia según tipo de sílaba e individuo")
plt.xlabel("Tipo")
plt.ylabel("Frecuencia (Hz)")
plt.legend(title="Individuo")
plt.show()

## }}}
## {{{ # Create a DataFrame from the data list para tipos de alfa

# Define the base path

base_path = "/home/ttdduu/lsd/tesis/datos/input/pitches_alfa_tipos"

# Initialize an empty list to store frequency values for each tipo and individual
data = []

# Define the types of individuals
individuals = ["A", "B", "19", "23", "34", "HEC1", "HAC1", "HAC2"]

for individual in individuals:
    # Construct the path to the individual's directory
    individual_path = os.path.join(base_path, individual)
    # Check if the directory exists
    if os.path.exists(individual_path):
        # Iterate through each directory (songs) in the individual's directory
        for songs_dir in os.listdir(individual_path):
            # Construct the path to the songs directory
            song_path = os.path.join(individual_path, songs_dir)
            # Check if it is a directory
            if os.path.isdir(song_path):
                types = os.listdir(song_path)
                for type in types:
                    type_path = os.path.join(song_path, type)
                    for file in os.listdir(type_path):
                        file_path = os.path.join(type_path, file)
                        if file_path.endswith(".txt"):
                            # Read the txt file and extract frequency values
                            with open(file_path, "r") as f:
                                # Extract frequency values from the second column
                                frequencies = process_syllable_file(f)
                                # Extend the data list with the tipo, individual, and frequencies
                                data.extend(
                                    [(individual, type, freq) for freq in frequencies]
                                )

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["Individuo", "Tipo", "Frecuencia"])

## }}}
## {{{ # Create a violin plot using seaborn para tipos de alfa

# Create a violin plot using seaborn
plt.figure(figsize=(12, 8))
sns.violinplot(
    data=df,
    x="Individuo",
    y="Frecuencia",
    hue="Tipo",
    # split=True,
    inner="box",
    palette="muted",
    inner_estimator="mean",
)
plt.title(
    "Distribución de valores de frecuencia según tipo de sílaba alfa en cada hornero hembra"
)
plt.xlabel("Individuo")
plt.ylabel("Frequency")
plt.legend(title="Tipo de sílaba")
plt.show()

## }}}
## {{{ hacer csv desde el df con el cual tmb hago el violin

grouped_data = (
    df.groupby(["Individuo", "Tipo"])["Frecuencia"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

# grouped_data.to_csv("frecuencia_stats.csv", index=False)
# a = grouped_data.T
# a.to_csv("frecuencia_stats_T.csv", index=False)

## }}}
