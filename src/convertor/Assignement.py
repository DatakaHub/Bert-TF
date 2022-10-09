from typing import Any, Dict
import tensorflow as tf


def assign_trained_weights(bert_model: tf.keras.Model, weights_path: str):
    loaded_original_model = tf.saved_model.load(weights_path)
    match_var_dict = {}
    for var in loaded_original_model.variables:
        match_var_dict[var.name] = var.numpy()
    for weight in bert_model.weights:
        if weight.name in match_var_dict:
            weight.assign(match_var_dict[weight.name])
        else:
            print(f"missing {weight.name}")


def get_bert_config(weights_path: str) -> Dict[str, Any]:
    loaded_original_model = tf.saved_model.load(weights_path)
    bert_config = {}
    num_attention_heads = 0
    for var in loaded_original_model.variables:
        if "word_embeddings" in var.name:
            bert_config["vocab_size"], bert_config["hidden_size"] = var.shape
        elif "/intermediate/kernel" in var.name:
            _, bert_config["intermediate_size"] = var.shape
        elif "type_embeddings" in var.name:
            bert_config["type_vocab_size"], _ = var.shape
        elif "transformer/layer_" in var.name:
            num_attention_heads = int(var.name.split("/")[1][6:]) + 1
    bert_config["initializer_range"] = 0.02
    bert_config["num_attention_heads"] = num_attention_heads
    return bert_config
