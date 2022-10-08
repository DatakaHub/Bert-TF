from typing import Callable, Any
import tensorflow as tf
from .Layers.OnDeviceEmbedding import create_OnDeviceEmbedding
from .Layers.PositionEmbedding import create_PositionEmbedding
from .Layers.SelfAttentionMask import create_SelfAttentionMask
from .Layers.TransformerEncoderBlock import create_TransformerEncoderBlock


def create_bert_encoder(
    vocab_size: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_attention_heads: int = 12,
    max_sequence_length: int = 512,
    type_vocab_size: int = 16,
    inner_dim: int = 3072,
    inner_activation: Callable = tf.keras.layers.Activation("gelu"),
    output_dropout: float = 0.1,
    attention_dropout: float = 0.1,
    initializer: tf.keras.initializers.Initializer = tf.keras.initializers.TruncatedNormal(
        stddev=0.02
    ),
    output_range: Any = None,
    embedding_width: Any = None,
    embedding_layer: Any = None,
    norm_first: bool = False,
    return_attention_scores: bool = False,
    **kwargs
) -> tf.keras.Model:
    """
    Bi-directional Transformer-based encoder network.
    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.
    The default values for this object are taken from the BERT-Base implementation
    in "BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding".
    This implementation is based on (https://github.com/tensorflow/models/blob/
    11b3662b3e3e1751a2b4285ea122c951796f2969/official/nlp/modeling/networks/
    bert_encoder.py#L318.)

    Args:
      vocab_size: The size of the token vocabulary.
      hidden_size: The size of the transformer hidden layers.
      num_layers: The number of transformer layers.
      num_attention_heads: The number of attention heads for each transformer. The
        hidden size must be divisible by the number of attention heads.
      max_sequence_length: The maximum sequence length that this encoder can
        consume. If None, max_sequence_length uses the value from sequence length.
        This determines the variable shape for positional embeddings.
      type_vocab_size: The number of types that the 'type_ids' input can take.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network for each transformer.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network for each transformer.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: The dropout rate to use for the attention layers within
        the transformer layers.
      initializer: The initialzer to use for all weights in this encoder.
      output_range: The sequence output range, [0, output_range), by slicing the
        target sequence of the last transformer layer. `None` means the entire
        target sequence will attend to the source sequence, which yields the full
        output.
      embedding_width: The width of the word embeddings. If the embedding width is
        not equal to hidden size, embedding parameters will be factorized into two
        matrices in the shape of ['vocab_size', 'embedding_width'] and
        ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
        smaller than 'hidden_size').
      embedding_layer: An optional Layer instance which will be called to generate
        embeddings for the input word IDs.
      norm_first: Whether to normalize inputs to attention and intermediate dense
        layers. If set False, output of attention and intermediate dense layers is
        normalized.
      dict_outputs: Whether to use a dictionary as the model outputs.
      return_all_encoder_outputs: Whether to output sequence embedding outputs of
        all encoder transformer layers. Note: when the following `dict_outputs`
        argument is True, all encoder outputs are always returned in the dict,
        keyed by `encoder_outputs`.
      return_attention_scores: Whether to add an additional output containing the
        attention scores of all transformer layers. This will be a list of length
        `num_layers`, and each element will be in the shape [batch_size,
        num_attention_heads, seq_dim, seq_dim].
    """

    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="input_word_ids"
    )
    mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="input_type_ids"
    )
    if embedding_width is None:
        embedding_width = hidden_size
    if embedding_layer is None:
        embedding_layer_inst = create_OnDeviceEmbedding(
            vocab_size=vocab_size,
            embedding_width=embedding_width,
            initializer=initializer,
            name="word_embeddings",
        )
    else:
        embedding_layer_inst = embedding_layer
    word_embeddings = embedding_layer_inst(word_ids)
    position_embedding_layer = create_PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name="position_embedding",
    )
    position_embeddings = position_embedding_layer(word_embeddings)
    type_embedding_layer = create_OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name="type_embeddings",
    )
    type_embeddings = type_embedding_layer(type_ids)
    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings]
    )
    embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32
    )
    embeddings = embedding_norm_layer(embeddings)
    embeddings = tf.keras.layers.Dropout(rate=output_dropout)(embeddings)
    if embedding_width != hidden_size:
        embedding_projection = tf.keras.layers.EinsumDense(
            "...x,xy->...y",
            output_shape=hidden_size,
            bias_axes="y",
            kernel_initializer=initializer,
            name="embedding_projection",
        )
        embeddings = embedding_projection(embeddings)
    else:
        embedding_projection = None

    transformer_layers = []
    data = embeddings
    SelfAttentionMask = create_SelfAttentionMask()
    attention_mask = SelfAttentionMask((data, mask))
    encoder_outputs = []
    attention_outputs = []
    for i in range(num_layers):
        transformer_output_range = None
        if i == num_layers - 1:
            transformer_output_range = output_range
        layer = create_TransformerEncoderBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            inner_dim=inner_dim,
            inner_activation=inner_activation,
            output_dropout=output_dropout,
            attention_dropout=attention_dropout,
            norm_first=norm_first,
            return_attention_scores=return_attention_scores,
            kernel_initializer=initializer,
            name="transformer/layer_%d" % i,
        )
        transformer_layers.append(layer)
        data = layer([data, attention_mask], output_range=transformer_output_range)
        if return_attention_scores:
            data, attention_scores = data
            attention_outputs.append(attention_scores)
        encoder_outputs.append(data)
    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation="tanh",
        kernel_initializer=initializer,
        name="pooler_transform",
    )
    cls_output = pooler_layer(first_token_tensor)

    outputs = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=cls_output,
        encoder_outputs=encoder_outputs,
    )
    if return_attention_scores:
        outputs['attention_scores'] = attention_outputs
        
    return tf.keras.Model(inputs=[word_ids, mask, type_ids], outputs=outputs)
