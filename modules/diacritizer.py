from queue import Queue
import keras
import tensorflow as tf
from utils import constants
from utils.preprocessor import Preprocessor


class Diacritizer:
    def __init__(
        self,
        servant="./servants/lstm-emb128-2rnn128-1dense256-v3",
        letters_tokenizer=None,
        diacritics_tokenizer=None,
    ):
        self.servant = tf.saved_model.load(servant)
        if letters_tokenizer is None:
            self.letters_tokenizer = self._get_default_letters_tokenizer()
        else:
            self.letters_tokenizer = letters_tokenizer
        if diacritics_tokenizer is None:
            self.diacritics_tokenizer = self._get_default_diac_tokenizer()
        else:
            self.diacritics_tokenizer = diacritics_tokenizer

        self.letters_w2i = keras.layers.StringLookup(
            vocabulary=self.letters_tokenizer.get_vocabulary(), mask_token=""
        )
        self.letters_i2w = keras.layers.StringLookup(
            vocabulary=self.letters_tokenizer.get_vocabulary(),
            mask_token="",
            invert=True,
        )

        self.diacritics_w2i = keras.layers.StringLookup(
            vocabulary=self.diacritics_tokenizer.get_vocabulary(), mask_token=""
        )
        self.diacritics_i2w = keras.layers.StringLookup(
            vocabulary=self.diacritics_tokenizer.get_vocabulary(),
            mask_token="",
            invert=True,
        )

    def __call__(self, text):
        return self.diacritize(text)

    def diacritize(self, text):
        splits = self.split_text(text)
        diacretized_text = self.join_and_diacritize_splits(splits)
        return diacretized_text

    def split_text(self, text):
        splits = Queue()
        text = Preprocessor.remove_tashkeel(text)
        idx = 0
        last_split_idx = 0
        while idx < len(text):
            if text[idx] not in constants.AR_LETTERS + " ":
                splits.put((text[last_split_idx:idx], "text"))
                splits.put((text[idx], "delimiter"))
                last_split_idx = idx + 1
            idx += 1
        if last_split_idx < len(text):
            splits.put((text[last_split_idx:], "text"))
        return splits

    def join_and_diacritize_splits(self, splits):
        res = ""
        while splits.qsize() > 0:
            split, split_type = splits.get()
            if split_type == "delimiter":
                res += split
            elif split_type == "text":
                res += self.infere_and_diacritize(split)
            else:
                raise ValueError(
                    "Unexpected split type value, possible values are (delimiter, text)"
                )
        return res

    def infere_and_diacritize(self, text):
        letters, _ = Preprocessor.strip_tashkeel(text)
        tokens = tf.cast(self.letters_tokenizer(letters)[tf.newaxis, ...], tf.float32)
        out = self.servant.serve(tokens)
        candidates = tf.argmax(out, -1)
        decoded_sentences = self.decode_sentences(tokens)
        decoded_diacritics = self.decode_diacritics(candidates)
        eof_idx = decoded_sentences[0].index("e")
        res = Preprocessor.combine_tashkeel(
            decoded_sentences[0][1:eof_idx], decoded_diacritics[0][1:eof_idx]
        )
        return res

    def decode_sentences(self, sentences):
        # work on batch size only
        return [
            [char.decode("utf-8") for char in l.numpy()]
            for l in self.letters_i2w(sentences)
        ]

    def decode_diacritics(self, diacritics):
        # work on batch size only
        return [
            [char.decode("utf-8") for char in d.numpy()]
            for d in self.diacritics_i2w(diacritics)
        ]

    def _get_default_letters_tokenizer(self):
        return keras.layers.TextVectorization(
            ragged=True,
            standardize=lambda x: tf.concat([["s"], x, ["e"]], axis=-1),
            split=None,
            vocabulary=constants.get_letters_vocabulary(),
        )

    def _get_default_diac_tokenizer(self):
        return keras.layers.TextVectorization(
            standardize=lambda x: tf.concat([[" "], x, [" "]], axis=-1),
            ragged=True,
            split=None,
            vocabulary=constants.get_diac_vocabulary(),
        )
