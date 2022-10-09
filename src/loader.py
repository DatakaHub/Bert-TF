from model.bert_architecture import create_bert_encoder
from convertor.Assignement import assign_trained_weights
import tensorflow as tf
import json


def load_bert_model(bert_config_path: str, weights_path: str) -> tf.keras.Model:
    """
    In order to create the Bert encoder model, we need some hyper-parameters
    which are stored in the config json file as well as the pre-trained weights.
    Such files come in the tar.gz files from tensorflow hub.

    Args:
        bert_config_path: path the hyper-parameter .json file
        weights_path: path the ckpt file with the wieght values
    """
    with open(bert_config_path, "r") as f:
        bert_config = json.load(f)

    bert_model = create_bert_encoder(
        vocab_size=bert_config["vocab_size"],
        hidden_size=bert_config["hidden_size"],
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config["initializer_range"]
        ),
        inner_dim=bert_config["intermediate_size"],
        num_attention_heads=bert_config["num_attention_heads"],
        type_vocab_size=bert_config["type_vocab_size"],
    )
    assign_trained_weights(bert_model, weights_path)
    return bert_model