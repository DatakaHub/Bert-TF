from bert_urls import map_name_to_handle
import os
import tarfile
from loader import load_bert_model
import tensorflow as tf


def retrieve_hub_data(bert_model_name: str) -> str:
    """
    This function downloads the bert files from tensorflow hub and uncompress them.
    To do so this function requires wget to be installed
    
    Args:
        bert_model_name: name of the bert model (has to be defined in bert_urls.py)
    """
    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
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
        os.chdir(
            os.path.join(os.path.dirname(current_path), "bert_data", bert_model_name)
        )
        tar = tarfile.open(f"{bert_model_name}.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        os.remove(f"{bert_model_name}.tar.gz")
    os.chdir(current_path)
    return os.path.join(
        os.path.dirname(current_path),
        "bert_data",
        bert_model_name,
    )


if __name__ == "__main__":
    for key in list(map_name_to_handle.keys()):
        print(f"extracting {key}")
        weights_path = retrieve_hub_data(bert_model_name=key)
        load_bert_model(
            weights_path=weights_path,
            save_path=os.path.join(
                os.path.dirname(os.getcwd()),
                "bert_data",
                f"{key}.h5",
            ),
        )
