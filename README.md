# BertKerasOnly


The goal of this project is to make an alternative solution to the current keras version of the Bert encoder, publically avaible. In this version , we don't use hub layers. Instead we only use standard keras layers for easier model manipulation.

This implementation uses tf.keras.layers.EinsumDense which are part of release tensorflow 2.10+.