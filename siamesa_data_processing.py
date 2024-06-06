# {{{ imports
import matplotlib.pyplot as plt
import os
from itertools import combinations
import copy
import itertools
import math
from PIL import Image
import numpy as np
import torch
import PIL
import random
import pickle

# }}}
# {{{ funcs: son todas las que uso en datos_splits


def load_images_as_tensors(filepaths):  # recibe una tupla de filepaths a la vez
    images = []
    for filepath in filepaths:
        # img = Image.open(filepath).convert("L").resize((240, 360))
        img = Image.open(filepath).convert("L").resize((194, 257))
        img_array = (np.array(img) == 255).astype("uint8")
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        images.append(img_tensor)
    return images


def lista_silabas_individuo(
    individual, base_path
):  # me genera la lista que va a ir en all_syllables_by_individual['indiv']:lista
    individual_path = os.path.join(base_path, individual)
    individual_list = []
    songs = os.listdir(individual_path)
    for song in songs:
        # CURRENTLY SI LA SONG TIENE NOMBRE "OBLIGATORIO", SUMAR TODAS. ELSE, AGARRAR UN SUBSET NOMÁS DE LAS SURROGADAS
        song_path = os.path.join(individual_path, song)
        png_files = [
            f"{song_path}/{file}"
            for file in os.listdir(song_path)
            if file.endswith(".png")
        ]
        tensors = load_images_as_tensors(png_files)  # los tensores de una canción
        individual_list.extend(tensors)
        random.shuffle(individual_list)  # agarro la lista de sílabas del individuo

    if len(individual_list)>80:
        return individual_list[:80]
    else:
        return individual_list


def listas_train_eval(
    individual_list, tr_split, max_train_samples=None
):  # con esta función me aseguro que ninguna de las sílabas de un set estén en otro
    train_point = int(
        np.round(tr_split * len(individual_list))
    )  # divido la lista de sílabas de este individuo (que randomicé en lista_silabas_individuo) con el valor tr_split
    lista_para_dict_train = individual_list[
        :train_point
    ]  # asigno cada parte a train o eval
    lista_para_dict_eval = individual_list[
        train_point:
    ]  # voy a particionar la lista "eval" en valid + test, o la dejo como test, lo defino en datos_splits
    if max_train_samples:
        # Transfer elements
        lista_para_dict_eval.extend(lista_para_dict_train[max_train_samples:])

        # Clear elements from lista_para_dict_train if you want
        del lista_para_dict_train[max_train_samples:]

    return lista_para_dict_train, lista_para_dict_eval


"""
esta func toma un diccionario que tiene keys y values de este modo: {'individuo':[lista de sílabas]}. Hay un diccionario para train y otro para valid o test. Returnea un diccionario:
{[tensor de sílaba 1, tensor de sílaba 2]:[0 o 1 si son mismo individuo o no, 'individuo de la 1era sílaba', 'individuo de la 2da sílaba']}
"""


def generar_combinaciones(set, template_dict):
    all_possible_combinations_by_individual = copy.deepcopy(template_dict)
    for individual in all_possible_combinations_by_individual.keys():
        all_possible_combinations_by_individual[individual] = list(
            list(itertools.combinations(set[1][individual], 2))
        )

    # unos
    result_dict = {}
    for key, value_list in all_possible_combinations_by_individual.items():
        for tensor_pair in value_list:
            result_dict[tensor_pair] = [1, key, key]
    all_possible_combinations_with_targets_one = result_dict

    total_elements = 0
    for value_list in all_possible_combinations_by_individual.values():
        total_elements += len(value_list)

    # ceros
    new_dict = {}
    original_dict = set[1]
    for key, tensor_list in original_dict.items():
        new_dict[key] = [[tensor, key] for tensor in tensor_list]
    lists = [i for i in new_dict.values()]  # es una lista de 5 listas
    combined_list = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            combinations = itertools.product(lists[i], lists[j])
            combined_list.extend(combinations)
    all_possible_combinations_with_targets_zero = {
        (item[0][0], item[1][0]): [0, item[0][1], item[1][1]] for item in combined_list
    }
    data_dict = all_possible_combinations_with_targets_one.copy()
    data_dict.update(all_possible_combinations_with_targets_zero)
    return data_dict


# }}}
# {{{ def datos_splits


"""
Con esta func genero los train, valid y test splits, y los guardo en una carpeta que especifico como arg (sets_path). La llamo antes de cada entrenamiento dentro de siamesa.py. De este modo, en siamesa.py genero y guardo los sets (la idea es randomizarlos antes de cada entrenamiento), e inmediatamente después los importo.

Lo más importante que recibe esta función es la carpeta pitches (con el nombre de argumento base_path), que son los archivos de texto con columnas de tiempo y frecuencia. El resto de los args son básicamente las carpetas donde guardar los sets creados.

Tengo ejemplos de cómo llamar esta función en siamesa.py, pero explico qué es cada argumento:

tr_specs tiene que ser una lista con esta estructura: ['nombre con el cual guardar el training set' , training split (por ej 0.8) del total de datos]

val_specs está defaulteado a None porque algunos datasets los quise generar con y otros sin un set de validación (por lo que dije en el informe: tengo pocos datos). De este modo, al no incluir un val_specs[1], tengo solo un split de train en 0.8 : test en 0.2.

sets_path es la carpeta donde guardar los sets de train y valid-test.

la lista individuals sirve porque esos son los nombres de las carpetas donde están las canciones de cada individuo; cada canción con sus sílabas, que son archivos txt, pasados a tensores por load_images_as_tensors.

"""


def datos_splits(
    tr_specs,
    test_name,
    base_path,
    individuals,
    sets_path,
    val_specs=None,
    individuals_unseen=[],
    max_train_samples=None,
):
    tr_name, tr_split = tr_specs[0], tr_specs[1]

    # inicio los dicts vacíos de {individuo:[lista de sílabas]} que voy a llenar con los pitches de cada individuo
    template_dict = {key: [] for key in individuals}
    template_dict_test = {key: [] for key in individuals + individuals_unseen}

    if val_specs:  # --> genero splits de valid y test
        val_name, val_split = val_specs[0], val_specs[1]
        (
            all_syllables_by_individual_train,
            all_syllables_by_individual_valid,
            all_syllables_by_individual_test,
        ) = (
            copy.deepcopy(template_dict),
            copy.deepcopy(template_dict),
            copy.deepcopy(template_dict),
        )

        for individual in individuals:
            individual_list = lista_silabas_individuo(
                individual, base_path
            )  # lista de todas las sílabas del indiv
            lista_para_dict_train = listas_train_eval(
                individual_list, tr_split, max_train_samples
            )[
                0
            ]  # el split de la lista para train
            all_syllables_by_individual_train[
                individual
            ] = lista_para_dict_train  # relleno la key del dict de train con la lista para train
            lista_para_dict_eval = listas_train_eval(
                individual_list, tr_split, max_train_samples
            )[
                1
            ]  # la voy a particionar en valid + test, o la dejo como test.
            valid_point = int(
                np.round(val_split * len(lista_para_dict_eval))
            )  # dentro de los datos de valid+test, valid es el 0.6
            lista_para_dict_valid = lista_para_dict_eval[:valid_point]
            all_syllables_by_individual_valid[individual] = lista_para_dict_valid
            lista_para_dict_test = lista_para_dict_eval[valid_point:]
            all_syllables_by_individual_test[individual] = lista_para_dict_test
            print(f"listatest = {len(lista_para_dict_test)}")
        train, valid, test = (
            all_syllables_by_individual_train,
            all_syllables_by_individual_valid,
            all_syllables_by_individual_test,
        )
        # guardar dict de test: {'individuo':[lista de sus sílabas]}. Todavía me queda generar todas las combinaciones posibles de los train y valid sets.
        with open(f"{sets_path}/{test_name}.pkl", "wb") as fp:
            pickle.dump(all_syllables_by_individual_test, fp)
            print("dictionary test saved successfully to file")

        # a partir de acá me falta generar todas las combinaciones posibles de sílabas dentro de cada set.

        sets = [[tr_name, train], [val_name, valid]]

    # si no quiero un set de validación, me armo solo los dicts de train y test

    # para cada individuo y cada uno de sus dicts (train-valid-test) relleno su key con la lista de sílabas que le toca

    if val_specs == None:
        (
            all_syllables_by_individual_train,
            all_syllables_by_individual_test,
        ) = copy.deepcopy(template_dict), copy.deepcopy(template_dict_test)
        for individual in individuals:
            individual_list = lista_silabas_individuo(
                individual, base_path
            )  # lista de todas las sílabas del indiv
            lista_para_dict_train = listas_train_eval(
                individual_list, tr_split, max_train_samples
            )[0]
            all_syllables_by_individual_train[individual] = lista_para_dict_train
            # acá abajo la func me devuelve eval y yo la llamo test
            lista_para_dict_test = listas_train_eval(
                individual_list, tr_split, max_train_samples
            )[
                1
            ]  # la voy a particionar en valid + test, o la dejo como test.
            all_syllables_by_individual_test[individual] = lista_para_dict_test

        # a cont relleno el dict de test en las keys de indivs no vistos
        if individuals_unseen:
            for individual in individuals_unseen:
                individual_list = lista_silabas_individuo(
                    individual, base_path
                )  # lista de todas las sílabas del indiv
                all_syllables_by_individual_test[individual] = lista_para_dict_test

        train, test = (
            all_syllables_by_individual_train,
            all_syllables_by_individual_test,
        )

        # guardar dict de test: {'individuo':[lista de sus sílabas]}. Todavía me queda generar todas las combinaciones posibles del train set.

        with open(f"{sets_path}/{test_name}.pkl", "wb") as fp:
            pickle.dump(all_syllables_by_individual_test, fp)
            print("dictionary test saved successfully to file")

        sets = [[tr_name, train]]

    # a partir de acá me falta generar todas las combinaciones posibles de sílabas dentro de cada set que haya generado.

    for set in sets:  #
        data_dict = generar_combinaciones(set, template_dict)
        print(f"largo del dict de train o valid: {len(data_dict)}")

        with open(f"{sets_path}/{set[0]}.pkl", "wb") as fp:
            pickle.dump(data_dict, fp)
            print("dictionary saved successfully to file")


# }}}
