import keras
from utils import constants


class RNNModel(keras.Model):
    def __init__(
        self,
        embedding_dims,
        rnn_type,
        rnn_layers,
        rnn_units,
        dense_layers,
        dense_units,
        dropout_rate=0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.rnn_type = rnn_type.upper()
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        self.embedding = keras.layers.Embedding(
            input_dim=len(constants.get_letters_vocabulary()) + 1,
            output_dim=embedding_dims,
            mask_zero=True,
        )
        self.rnn = [
            (
                keras.layers.Bidirectional(
                    keras.layers.LSTM(
                        rnn_units,
                        return_sequences=True,
                        dropout=self.dropout_rate,
                    )
                )
                if self.rnn_type == "LSTM"
                else keras.layers.Bidirectional(
                    keras.layers.GRU(
                        rnn_units,
                        return_sequences=True,
                        dropout=self.dropout_rate,
                    )
                )
            )
            for _ in range(rnn_layers)
        ]
        self.dense = [
            keras.layers.Dense(dense_units, activation="relu")
            for _ in range(dense_layers)
        ]
        self.output_layer = keras.layers.Dense(len(constants.get_diac_vocabulary()) + 1)

    def call(self, inputs, training=False):
        x = self.embedding(inputs, training=training)

        for rnn_layer in self.rnn:
            x = rnn_layer(x, training=training)

        for dense_layer in self.dense:
            x = dense_layer(x, training=training)
        x = self.output_layer(x, training=training)
        return x

    def build(self, input_shape):
        out_shape = input_shape
        self.embedding.build(out_shape)
        out_shape = self.embedding.compute_output_shape(out_shape)

        for rnn_layer in self.rnn:
            rnn_layer.build(out_shape)
            out_shape = rnn_layer.compute_output_shape(out_shape)

        for dense_layer in self.dense:
            dense_layer.build(out_shape)
            out_shape = dense_layer.compute_output_shape(out_shape)
        self.output_layer.build(out_shape)
