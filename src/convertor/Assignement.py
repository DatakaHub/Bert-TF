import tensorflow as tf


def assign_trained_weights(bert_model:tf.keras.Model, weights_path:str):
    for weight in bert_model.weights:
        print(weight.shape, weight.name)