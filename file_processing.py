import json


def save_to_json(data_series, filename:str):
    """Create json file from dataframe column."""
    hash_map = {}

    for index, value in enumerate(data_series):
        hash_map[index] = value

    try:
        with open(filename, "w") as json_file:
            json.dump(obj=hash_map, fp=json_file, indent=4)
    except PermissionError as perm_err:
        print(f"PermissionError: {perm_err.args}")
    except IOError as io_err:
        print(f"IOError: {io_err.args}")


def load_from_json(filename:str):
    """Read json into hash map."""
    try:
        with open(filename, "r") as json_file:
            return json.load(fp=json_file)
    except FileNotFoundError as missing_err:
        print(f"File not found: {missing_err.args}")
    except json.JSONDecodeError as decoding_err:
        print(f"Invalid JSON format: {decoding_err.args}")