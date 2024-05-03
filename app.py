import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from modules.diacritizer import Diacritizer

app = Flask(__name__)
api = Api(app)

diacritizer = Diacritizer()


class Shakkel(Resource):
    def post(self):
        text = request.json["text"]
        return {"diacritized": diacritizer.diacritize(text)}


api.add_resource(Shakkel, "/shakkel")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
