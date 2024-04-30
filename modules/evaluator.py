import tensorflow as tf
from tqdm import tqdm
from utils.preprocessor import Preprocessor
from diacritization_evaluation import wer, der
import os


class Evaluator:
    def __init__(self, diacritizer):
        self.diacritizer = diacritizer

    def calculate_metrics(self, ds, output_path="./results/wer-der.txt"):
        # Works on batched datasets only
        def combine_per_sen(sen, diac):
            eof_idx = sen.index("e")
            res = Preprocessor.combine_tashkeel(sen[1:eof_idx], diac[1:eof_idx])
            return res

        ds_len = len(ds) * next(iter(ds.take(10)))[0].shape[0]
        res = {"wer": 0, "wer*": 0, "der": 0, "der*": 0}
        for sen, diac in tqdm(ds, desc="Evaluating..."):
            sen = tf.cast(sen, tf.float32)
            preds = self.diacritizer.servant.serve(sen)
            decoded_sentences = self.diacritizer.decode_sentences(sen)
            decoded_diacritics_t = self.diacritizer.decode_diacritics(diac)
            decoded_diacritics_p = self.diacritizer.decode_diacritics(
                tf.argmax(preds, -1)
            )
            idx = 0
            while idx < len(decoded_sentences):
                true_diac = combine_per_sen(
                    decoded_sentences[idx], decoded_diacritics_t[idx]
                )
                pred_diac = combine_per_sen(
                    decoded_sentences[idx], decoded_diacritics_p[idx]
                )
                res["wer"] += wer.calculate_wer(true_diac, pred_diac) / ds_len
                res["wer*"] += (
                    wer.calculate_wer(true_diac, pred_diac, case_ending=False) / ds_len
                )
                res["der"] += der.calculate_der(true_diac, pred_diac) / ds_len
                res["der*"] += (
                    der.calculate_der(true_diac, pred_diac, case_ending=False) / ds_len
                )
                idx += 1

        with open(output_path, "w+") as f:
            f.write(f"WER with case ending: {res['wer']}\n")
            f.write(f"WER without case ending: {res['wer*']}\n")
            f.write(f"DER with case ending: {res['der']}\n")
            f.write(f"DER without case ending: {res['der*']}\n")
            f.seek(0)
            print(f.read())

        return res
