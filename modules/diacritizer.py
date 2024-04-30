import keras


class Diacritizer:
    def __init__(
        self,
        model,
        letters_tokenizer,
        diacritics_tokenizer,
    ):
        self.model = model
        self.letters_w2i = keras.layers.StringLookup(
            vocabulary=letters_tokenizer.get_vocabulary(), mask_token=""
        )
        self.letters_i2w = keras.layers.StringLookup(
            vocabulary=letters_tokenizer.get_vocabulary(), mask_token="", invert=True
        )

        self.diacritics_w2i = keras.layers.StringLookup(
            vocabulary=diacritics_tokenizer.get_vocabulary(), mask_token=""
        )
        self.diacritics_i2w = keras.layers.StringLookup(
            vocabulary=diacritics_tokenizer.get_vocabulary(), mask_token="", invert=True
        )

    def diacritize(self, sentence):
        pass

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
