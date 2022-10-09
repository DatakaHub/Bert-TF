from typing import Callable, Tuple
import tensorflow as tf


def create_SelfAttentionMask() -> Callable:
    """
    based on https://github.com/tensorflow/models/blob/11b3662b3e3e1751a2b4285ea122c951796f2969/official/nlp/modeling/layers/self_attention_mask.py#L49
    """

    def SelfAttentionMask(inputs: Tuple[tf.Variable, tf.Variable]) -> tf.Variable:
        to_mask = inputs[1]
        inputs = inputs[0]
        from_shape = tf.shape(inputs)
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        dtype = inputs.dtype
        to_shape = tf.shape(to_mask)
        to_seq_length = to_shape[1]
        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype
        )
        return tf.broadcast_to(to_mask, [batch_size, from_seq_length, to_seq_length])

    return SelfAttentionMask
