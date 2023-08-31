import logging

import numpy as np
from os import path, makedirs
import json
import pickle


def iterations_finder(_path):
    i = 1
    split = path.splitext(_path)
    is_dir = not split[1]
    operation = path.isdir if is_dir else path.isfile
    while operation(split[0] + str(i) + split[1]):
        i += 1
    return split, i


def find_latest_path_iterations(_path):
    split, i = iterations_finder(_path)
    return split[0] + str(max(1, i - 1)) + split[1]


def find_next_path_iterations(_path):
    split, i = iterations_finder(_path)
    return split[0] + str(i) + split[1]


def save_points_cloud(points, filename, out_dir='/tmp'):
    # if OUT_DIR is not a _path, create it
    if not path.isdir(out_dir):
        makedirs(out_dir)
    np.savetxt(find_next_path_iterations(out_dir + "/" + filename + '.xyz'), points)


saving_vars = [
    "sets_per_frame_1",
    "sets_per_frame_2",
    "desc_dict_1",
    "desc_dict_2"
]


def loadSystemState(out_dir):
    loaded_variables = load_variables_by_dict(saving_vars, out_dir)
    tup = ()
    for key in saving_vars:
        tup += (loaded_variables[key],)
    return tup


def saveSystemState(vars_dict, out_dir):
    if len(saving_vars) != len(vars_dict):
        raise ValueError("Number of variables to save does not match number of arguments needed!")

    variables_to_save = {}
    for i in range(len(saving_vars)):
        variables_to_save[saving_vars[i]] = vars_dict[i]
    out_dir = find_next_path_iterations(out_dir + "/iteration")
    save_variables_by_dict(variables_to_save, out_dir)


def load_camera_data_json(json_path, dtype=np.float32):
    with open(json_path, "r") as file:
        data = json.load(file)
    return np.array(data['k'], dtype=dtype), np.array(data['d'], dtype=dtype), np.array(data['dims'], dtype=np.int32)


def save_variables_by_dict(variables, out_dir="/tmp"):
    # Gets a dictionary of variables and saves them to separate files
    if not path.isdir(out_dir):
        makedirs(out_dir)
    for key, value in variables.items():
        with open(out_dir + "/" + key + '.pkl', 'wb') as f:
            pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)


def load_variables_by_dict(variables_names, directory):
    # Gets a dictionary of variables names and returns a dictionary of the variables
    variables = {}
    for key in variables_names:
        try:
            with open(directory + "/" + key + '.pkl', 'rb') as f:
                variables[key] = pickle.load(f)
        except FileNotFoundError:
            logging.exception("File not found: " + directory + "/" + key + '.pkl')
            variables[key] = None
    return variables


