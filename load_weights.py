import keras
import tensorflow as tf
import os
import test_set_from_pkl
import plotter as p
import pickle
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def opd(path):
    with open(path, "rb") as fp:
        dict_name = pickle.load(fp)
    return dict_name


def load(path_weights, test_pkl):
    test_dict = opd(test_pkl)

    test_set, test_tensors_belong_to = test_set_from_pkl.make_test_set(test_dict)
    base_network = keras.models.load_model(path_weights, custom_objects={"tf": tf})
    return test_set, test_tensors_belong_to, base_network
