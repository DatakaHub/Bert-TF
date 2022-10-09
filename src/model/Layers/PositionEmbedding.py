from typing import Callable
import tensorflow as tf


def create_PositionEmbedding(
    initializer: tf.keras.initializers.Initializer,
    max_length: int,
    name: str = "position_embedding",
) -> Callable:
    """
    base on https://github.com/tensorflow/models/blob/11b3662b3e3e1751a2b4285ea122c951796f2969/official/nlp/modeling/layers/position_embedding.py#L28
    """

    def PositionEmbedding(inputs: tf.Variable) -> tf.Variable:
        embeddings = tf.Variable(
            initial_value=initializer(shape=(max_length, inputs.shape[-1])),
            trainable=True,
            name=name,
        )
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[1]
        position_embeddings = embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in inputs.get_shape().as_list()]
        new_shape[1] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)

    return PositionEmbedding
