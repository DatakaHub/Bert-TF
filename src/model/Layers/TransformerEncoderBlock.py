from typing import Callable, Any, Optional
import tensorflow as tf


def create_TransformerEncoderBlock(
    hidden_size: int,
    num_attention_heads: int,
    inner_dim: int,
    inner_activation: tf.keras.layers.Activation,
    output_dropout: float,
    attention_dropout: float,
    norm_first: bool,
    return_attention_scores: bool,
    kernel_initializer: tf.keras.initializers.Initializer,
    name: str,
) -> Callable:
    common_kwargs = dict(
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )
    einsum_equation = "abc,cd->abd"
    _inner_dim = inner_dim
    _output_range = None
    _norm_first = norm_first
    _norm_epsilon = (1e-12,)
    _attention_layer_norm = tf.keras.layers.LayerNormalization(
        name=f"{name}/self_attention_layer_norm",
        axis=-1,
        epsilon=_norm_epsilon,
        dtype=tf.float32,
    )
    _attention_layer_norm_kv = _attention_layer_norm
    _diff_q_kv_att_layer_norm = False
    if _diff_q_kv_att_layer_norm:
        _attention_layer_norm_kv = tf.keras.layers.LayerNormalization(
            name=f"{name}/self_attention_layer_norm_kv",
            axis=-1,
            epsilon=_norm_epsilon,
            dtype=tf.float32,
        )
    _return_attention_scores = return_attention_scores
    _attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=int(hidden_size // num_attention_heads),
        value_dim=None,
        dropout=attention_dropout,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer="zeros",
        attention_axes=None,
        output_shape=None,
        name=f"{name}/self_attention",
    )
    _attention_dropout_rate = attention_dropout
    _attention_dropout = tf.keras.layers.Dropout(rate=_attention_dropout_rate)
    _use_query_residual = True
    _output_layer_norm = tf.keras.layers.LayerNormalization(
        name=f"{name}/output_layer_norm",
        axis=-1,
        epsilon=_norm_epsilon,
        dtype=tf.float32,
    )
    _intermediate_dense = tf.keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, _inner_dim),
        bias_axes="d",
        kernel_initializer=kernel_initializer,
        bias_initializer="zeros",
        name=f"{name}/intermediate",
        **common_kwargs,
    )
    _inner_dropout = 0.0
    _inner_dropout_layer = tf.keras.layers.Dropout(rate=_inner_dropout)
    last_output_shape = hidden_size
    _output_dense = tf.keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, last_output_shape),
        bias_axes="d",
        name=f"{name}/output",
        kernel_initializer=kernel_initializer,
        bias_initializer="zeros",
        **common_kwargs,
    )
    _output_dropout = tf.keras.layers.Dropout(rate=output_dropout)

    def TransformerEncoderBlock(
        inputs: Any, output_range: Optional[tf.Tensor] = None
    ) -> Any:
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                input_tensor, attention_mask = inputs
                key_value = None

        if output_range is None:
            output_range = _output_range
        if output_range:
            if _norm_first:
                source_tensor = input_tensor[:, 0:output_range, :]
                input_tensor = _attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = _attention_layer_norm_kv(key_value)
            target_tensor = input_tensor[:, 0:output_range, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0:output_range, :]
        else:
            if _norm_first:
                source_tensor = input_tensor
                input_tensor = _attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = _attention_layer_norm_kv(key_value)
            target_tensor = input_tensor

        if key_value is None:
            key_value = input_tensor

        if _return_attention_scores:
            attention_output, attention_scores = _attention_layer(
                query=target_tensor,
                value=key_value,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )
        else:
            attention_output = _attention_layer(
                query=target_tensor, value=key_value, attention_mask=attention_mask
            )
        attention_output = _attention_dropout(attention_output)

        if _norm_first:
            if _use_query_residual:
                attention_output = source_tensor + attention_output
        else:
            if _use_query_residual:
                attention_output = target_tensor + attention_output
                attention_output = _attention_layer_norm(attention_output)

        if _norm_first:
            source_attention_output = attention_output
            attention_output = _output_layer_norm(attention_output)
        inner_output = _intermediate_dense(attention_output)
        inner_output = inner_activation(inner_output)
        inner_output = _inner_dropout_layer(inner_output)
        layer_output = _output_dense(inner_output)
        layer_output = _output_dropout(layer_output)

        if _norm_first:
            layer_output = source_attention_output + layer_output
        else:
            layer_output = tf.cast(layer_output, tf.float32)
            layer_output = _output_layer_norm(layer_output + attention_output)

        if _return_attention_scores:
            return layer_output, attention_scores
        else:
            return layer_output

    return TransformerEncoderBlock
