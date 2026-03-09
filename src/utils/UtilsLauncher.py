import json
import os
from datetime import datetime

def json_reader(path_json_file):
    with open(path_json_file, "r") as f:
        config = json.load(f)
    return config

def data_config_reader(config):
    data_config = config["data"]
    data_dir = data_config["data_dir"]
    train_instances = data_config["train_instances"]
    validation_instances =data_config["validation_instances"]
    evaluation_instances = data_config["evaluation_instances"]
    return data_dir, train_instances, validation_instances, evaluation_instances

def add_dated_folder(base_path):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    new_path = os.path.join(base_path, date_str)
    index = 1
    while os.path.exists(new_path):
        new_path = os.path.join(base_path, f"{date_str}_{index}")
        index += 1
    os.makedirs(new_path, exist_ok=True)
    return new_path

def json_saver(output_dir, config):
    config_save_path = os.path.join(output_dir, "config_used.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration sauvegardée dans : {config_save_path}")