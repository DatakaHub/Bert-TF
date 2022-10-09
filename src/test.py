from bert_urls import map_name_to_handle

bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
tfhub_handle_encoder = map_name_to_handle[bert_model_name]

import os
import requests
import tarfile

current_path = os.getcwd()
print()
if not os.path.exists(os.path.join(os.path.dirname(current_path), "bert_data")):
    os.mkdir(os.path.join(os.path.dirname(current_path), "bert_data"))

os.chdir(os.path.join(os.path.dirname(current_path), "bert_data"))
initial_list = os.listdir()
if bert_model_name not in os.listdir():
    os.system(f'wget "{tfhub_handle_encoder}"')
    downloaded_file = list(set(os.listdir()) - set(initial_list))[0]
    os.mkdir(bert_model_name)
    os.rename(
        downloaded_file, os.path.join(bert_model_name, f"{bert_model_name}.tar.gz")
    )
    os.chdir(os.path.join(os.path.dirname(current_path), "bert_data", bert_model_name))
    tar = tarfile.open(f"{bert_model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()
    os.remove(f"{bert_model_name}.tar.gz")
os.chdir(current_path)

import json
import re

bert_config_path = "bert_config.json"

with open(bert_config_path, "r") as f:
    bert_config = json.load(f)
    bert_config["hidden_size"] = int(re.findall("\d+", bert_model_name)[0])
    bert_config["num_attention_heads"] = int(re.findall("\d+", bert_model_name)[1])
    bert_config["num_hidden_layers"] = int(re.findall("\d+", bert_model_name)[2])

from loader import load_bert_model

bert_model = load_bert_model(
    bert_config_path=bert_config_path,
    weights_path=os.path.join(
        os.path.dirname(current_path),
        "bert_data",
        bert_model_name,
        "saved_model.pb"
    ),
)
