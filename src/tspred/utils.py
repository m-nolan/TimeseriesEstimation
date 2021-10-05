"""utils.py

Utility functions supporting TimeseriesPrediction model code.

"""

# utils.py

import os
import re

def create_model_pathstr(hparams):
    """create_model_pathstr

    Creates a unique model directory string from a given hyperparameter file.
    Any two identical hyperparameter YAML files will produce the same pathstr. This will group multiple models with the same hyperparameter configuration.

    NOTE: Proper use is highly dependent on the structure of the YAML file. The following class information order is required for an organized experiment directory:
    - datamodule
    - model
    - objective
    - optimizer
    - trainer

    Args:
        hparams (dict): Hyperparameter dict object read from a project YAML file.

    Returns:
        model_path (str): Model path string, unique for a given hyperparameterization
    """

    dir_string_list = []
    for class_key, class_dict in hparams.items():
        dir_string = []
        dir_string.append(class_key)
        for cd_key, cd_val in class_dict.items():
            if isinstance(cd_val, (list, tuple)):
                continue
            if isinstance(cd_val, dict):
                for cdd_key, cdd_val in cd_val.items():
                    cat_string = cd_key+'_'+cdd_key
                    dir_string.append(f'{snake2camel(cat_string)}-{cdd_val}')
            else:
                dir_string.append(f'{snake2camel(cd_key)}-{cd_val}')
        
        dir_string = '_'.join(dir_string)
        print(dir_string)
        dir_string_list.append(dir_string)
    model_path = os.path.join(*dir_string_list)

    return model_path

def snake2camel(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel2snake(camel_str):
    components = re.findall('^[a-z]+|[A-Z][^A-Z]*', camel_str)
    return '_'.join(c.lower() for c in components)
