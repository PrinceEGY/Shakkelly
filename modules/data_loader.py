import pandas as pd
import tensorflow as tf

from utils.preprocessor import Preprocessor


class DataLoader:
    def __init__(self, letter_tokenizer, diac_tokenizer):
        self.letter_tokenizer = letter_tokenizer
        self.diac_tokenizer = diac_tokenizer

    def from_csv(self, file_path):
        # TODO: replace reading files using pandas to reduce loading time
        df = pd.read_csv(file_path)
        return tf.data.Dataset.from_tensor_slices(df["text"])

    def process_ds(self, ds, batch_size=32, shuffle_buffer=1000):
        ds = (
            ds.map(self.tf_strip_tashkeel, tf.data.AUTOTUNE)
            .map(
                lambda x, y: (self.letter_tokenizer(x), self.diac_tokenizer(y)),
                tf.data.AUTOTUNE,
            )
            .shuffle(shuffle_buffer)
            .padded_batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return ds

    def tf_strip_tashkeel(self, inputs):
        @tf.py_function(Tout=(tf.string, tf.string))
        def strip_tashkeel(inputs):
            text = inputs.numpy().decode("utf-8")
            text, tashkeel = Preprocessor.strip_tashkeel(text)
            text = tf.convert_to_tensor(text, dtype=tf.string)
            tashkeel = tf.convert_to_tensor(tashkeel, dtype=tf.string)
            return text, tashkeel

        text, tashkeel = strip_tashkeel(inputs)
        text.set_shape((None,))
        tashkeel.set_shape((None,))
        return text, tashkeel

    def merge_datasets(self, datasets, shuffle_buffer=1000):
        ds = datasets[0]
        for d in datasets[1:]:
            ds = ds.concatenate(d)
        return ds.shuffle(shuffle_buffer)
