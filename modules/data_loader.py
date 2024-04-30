import pandas as pd
import tensorflow as tf

from utils.preprocessor import Preprocessor

class DataLoader:
    def __init__(self, letter_tokenizer, diac_tokenizer):
        self.letter_tokenizer = letter_tokenizer
        self.diac_tokenizer = diac_tokenizer
    
    def from_csv(self, file_path, only_text=True):
        ds = tf.data.experimental.make_csv_dataset(file_path, batch_size=1)
        if only_text:
            ds = ds.map(lambda x: x["text"])
        return ds.unbatch()
    
    def process_ds(self, ds, batch_size=32, shuffle_buffer=1000):
        ds = (
        ds.map(self.tf_strip_tashkeel, tf.data.AUTOTUNE)
        .map(lambda x, y: (self.letter_tokenizer(x), self.diac_tokenizer(y)), tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer)
        .padded_batch(batch_size)
        )
    
        return ds

    def tf_strip_tashkeel(self, inputs):
        @tf.py_function(Tout=(tf.string, tf.string))
        def strip_tashkeel(inputs):
            text = inputs.numpy().decode('utf-8')
            text, tashkeel = Preprocessor.strip_tashkeel(text)
            text = tf.convert_to_tensor(text, dtype=tf.string)
            tashkeel = tf.convert_to_tensor(tashkeel, dtype=tf.string)
            return text, tashkeel
        text, tashkeel = strip_tashkeel(inputs)
        text.set_shape((None, ))
        tashkeel.set_shape((None, ))
        return text, tashkeel