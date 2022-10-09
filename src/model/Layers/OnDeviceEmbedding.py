from typing import Any, Callable, Dict, Type
import tensorflow as tf
import numpy as np


class OnDeviceEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, 
            vocab_size: int, 
            initializer: tf.keras.initializers.Initializer, 
            use_one_hot:bool,
            embedding_width:int,
            trainable:bool=True, 
            name:str=None,
            dtype:Type=None, 
            dynamic:bool=False, 
            **kwargs:Any):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.vocab_size=vocab_size
        self.initializer=initializer
        self.use_one_hot=use_one_hot
        self.embedding_width=embedding_width
        
    def build(self, input_shape:tuple):
        self.embeddings_var = tf.Variable(
            initial_value=self.initializer(shape=(self.vocab_size, self.embedding_width)),
            trainable=True,
            name="embeddings",
        )
        
    def call(self, inputs, training=None, *args, **kwargs):
        if training is not None:
            training = tf.cast(x=training, dtype=tf.bool)
        else:
            training = tf.cast(x=tf.keras.backend.learning_phase(), dtype=tf.bool)
        flat_inputs = tf.reshape(inputs, [-1])
        if self.use_one_hot:
            dtype = tf.float32
            one_hot_data = tf.one_hot(flat_inputs, depth=self.vocab_size, dtype=dtype)
            embeddings = tf.matmul(one_hot_data, self.embeddings_var)
        else:
            embeddings = tf.gather(self.embeddings_var, flat_inputs)
        embeddings = tf.reshape(
            embeddings, tf.concat([tf.shape(inputs), [self.embedding_width]], axis=0)
        )
        embeddings.set_shape(inputs.shape.as_list() + [self.embedding_width])
        return embeddings
    
    def get_config(self)->Dict[str,Any]:
        config = super().get_config()
        config["vocab_size"]=self.vocab_size
        config["initializer"]=self.initializer
        config["use_one_hot"]=self.use_one_hot
        config["embedding_width"]=self.embedding_width
        config["trainable"]=self.trainable
        config["name"]=self.name
        config["dtype"]=self.dtype
        config["dynamic"]=self.dynamic
        return config

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

    return OnDeviceEmbeddingLayer(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        name=name,
        use_one_hot=use_one_hot)
