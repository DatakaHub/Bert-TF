# BertKerasOnly


The goal of this project is to make an alternative solution to the current keras version of the Bert encoder, publically avaible. In this version, we don't use hub layers. Instead we only use standard keras layers for easier model manipulation.

This implementation uses tf.keras.layers.EinsumDense which are part of release tensorflow 2.10+.

## Tutorial

You can use the [loading tutorial](src/LoadingExample.ipynb) on how to load and convert a bert mdoel. You will need tensorflow 2.10+ and wget.

## How to use

If you want to extract a bert once and use in another script, you will need some custom objects as these are not saved in .h5 file but are required a loading time. You will need to import the [OnDeviceEmbedding layer](src/model/Layers/OnDeviceEmbedding.py) as well as the [PositionEmbedding layer](src/model/Layers/PositionEmbedding.py) before the call the `tf.keras.models.load_model(). Another option consists in duplciating the two classes in your own code instead of importing this project.

### CLI

To extract all bert models :
```
python -m bert_tf_extract --model_name all
```

To extract a specific bert model :
```
python -m bert_tf_extract --model_name bert_en_wwm_uncased_L-24_H-1024_A-16 --saveh5 /path/to/my_bert_tf.h5
```

## How to infer with Bert

This project aims at converting the Bert encoders to a more easy to use tf.keras.Model. In order to infer, one will need the adequate tokenizer as explained in [this tensorflow tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).
