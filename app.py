from flask import Flask, request
from flask_restful import Resource, Api
import keras
import tensorflow as tf
from modules.diacritizer import Diacritizer
from utils import constants

app = Flask(__name__)
api = Api(app)
letters_tok = keras.layers.TextVectorization(
    ragged=True,
    standardize=lambda x: tf.concat([["s"], x, ["e"]], axis=-1),
    split=None,
    vocabulary=constants.get_letters_vocabulary(),
)

diac_tok = keras.layers.TextVectorization(
    standardize=lambda x: tf.concat([[" "], x, [" "]], axis=-1),
    ragged=True,
    split=None,
    vocabulary=constants.get_diac_vocabulary(),
)
diacritizer = Diacritizer(
    servant="./servants/gru-emb128-2rnn128-1dense256",
    letters_tokenizer=letters_tok,
    diacritics_tokenizer=diac_tok,
)


class Shakkel(Resource):
    def post(self):
        text = request.json["text"]
        return {"diacritized": diacritizer.diacritize(text)}


api.add_resource(Shakkel, "/shakkel")
if __name__ == "__main__":
    app.run(debug=True, port=5000)