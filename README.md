# BertKerasOnly


The goal of this project is to make an alternative solution to the current keras version of the Bert encoder, publically avaible. In this version, we don't use hub layers. Instead we only use standard keras layers for easier model manipulation.

This implementation uses tf.keras.layers.EinsumDense which are part of release tensorflow 2.10+.

## Tutorial

You can use the [loading tutorial](src/LoadingExample.ipynb) on how to load and convert a bert mdoel. You will ned tensorflow 2.10+ and wget.

## How to use

If you want to extract a bert once and use in another script, you will need some custom objects as these are not saved in .h5 file but are required a loading time. You will need to import the [OnDeviceEmbedding layer](src/model/Layers/OnDeviceEmbedding.py) as well as the [PositionEmbedding layer](src/model/Layers/PositionEmbedding.py) before the call the `tf.keras.models.load_model(). Another option consists in duplciating the two classes in your own code instead of importing this project.