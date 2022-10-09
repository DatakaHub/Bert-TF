from typing import Any, Callable, Dict, Type
import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
            max_length: int, 
            initializer: tf.keras.initializers.Initializer, 
            trainable:bool=True, 
            name:str=None,
            dtype:Type=None, 
            dynamic:bool=False, 
            **kwargs:Any):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.max_length=max_length
        self.initializer=initializer
        
    def build(self, input_shape:tuple):
        self.embeddings = tf.Variable(
            initial_value=self.initializer(shape=(self.max_length, input_shape[-1])),
            trainable=True,
            name=self.name,
        )
        
    def call(self, inputs, training=None, *args, **kwargs):
        if training is not None:
            training = tf.cast(x=training, dtype=tf.bool)
        else:
            training = tf.cast(x=tf.keras.backend.learning_phase(), dtype=tf.bool)
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[1]
        position_embeddings =self.embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in inputs.get_shape().as_list()]
        new_shape[1] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)
    
    def get_config(self)->Dict[str,Any]:
        config = super().get_config()
        config["max_length"]=self.max_length
        config["initializer"]=self.initializer
        config["trainable"]=self.trainable
        config["name"]=self.name
        config["dtype"]=self.dtype
        config["dynamic"]=self.dynamic
        return config


def create_PositionEmbedding(
    initializer: tf.keras.initializers.Initializer,
    max_length: int,
    name: str = "position_embedding",
) -> Callable:
    """
    base on https://github.com/tensorflow/models/blob/11b3662b3e3e1751a2b4285ea122c951796f2969/official/nlp/modeling/layers/position_embedding.py#L28
    """

    return PositionEmbedding(max_length=max_length, name=name, initializer=initializer)