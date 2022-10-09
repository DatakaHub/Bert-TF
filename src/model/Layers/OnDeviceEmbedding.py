from typing import Callable
import tensorflow as tf
import numpy as np


def create_OnDeviceEmbedding(
    vocab_size: int,
    embedding_width: int,
    initializer: tf.keras.initializers.Initializer,
    name: str = "word_embeddings",
    use_one_hot: bool = False,
) -> Callable:
    """
    based on https://github.com/tensorflow/models/blob/11b3662b3e3e1751a2b4285ea122c951796f2969/official/nlp/modeling/layers/on_device_embedding.py#L22
    """

    def OnDeviceEmbeddingLayer(inputs: tf.Variable) -> tf.Variable:
        embeddings_var = tf.Variable(
            initial_value=initializer(shape=(vocab_size, embedding_width)),
            trainable=True,
            name=name,
        )
        flat_inputs = tf.reshape(inputs, [-1])
        if use_one_hot:
            dtype = tf.float32
            one_hot_data = tf.one_hot(flat_inputs, depth=vocab_size, dtype=dtype)
            embeddings = tf.matmul(one_hot_data, embeddings_var)
        else:
            embeddings = tf.gather(embeddings_var, flat_inputs)
        embeddings = tf.reshape(
            embeddings, tf.concat([tf.shape(inputs), [embedding_width]], axis=0)
        )
        embeddings.set_shape(inputs.shape.as_list() + [embedding_width])
        return embeddings

    return OnDeviceEmbeddingLayer
