import matplotlib.pyplot as plt
import keras
import os

keras.utils.set_random_seed(2024)


class Trainer:
    def __init__(
        self,
        model,
        name,
        optimizer,
        loss_fn,
        train_ds,
        valid_ds,
        epochs,
        steps_per_epoch,
        validation_steps,
        callbacks,
        verbose=1,
        weights_save_dir="./weights/",
        servant_save_dir="./servants/",
    ):
        self.model = model
        self.name = name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self.verbose = verbose
        self.weights_save_dir = weights_save_dir
        self.servant_save_dir = servant_save_dir

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        history = self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        loss_path = os.path.join("results", self.name)
        self._save_loss_plot(history, save_path=loss_path)

    def save(self, save_weights_only=True):
        os.makedirs(self.weights_save_dir, exist_ok=True)
        path = os.path.join(
            self.weights_save_dir, self._parse_name(self.name, save_weights_only)
        )
        if save_weights_only:
            self.model.save_weights(path)
        else:
            self.model.save(path)
        print(f"Model saved at {path}")

    def save_servant(self):
        os.makedirs(self.servant_save_dir, exist_ok=True)
        path = os.path.join(self.servant_save_dir, self.name)
        self.model.export(path)
        print(f"Servant saved at {path}")

    def _save_loss_plot(self, history, save_path="./results/"):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel("Epoch #")
        plt.ylabel("Cross Entropy/token")
        plt.legend()
        plt.savefig(save_path + "-loss_plot.png")

    def _parse_name(self, save_dir: str, save_weights_only: bool) -> str:
        if save_weights_only:
            if save_dir.endswith(".weights.h5"):
                return save_dir
            elif save_dir.endswith(".h5"):
                return save_dir[:-3] + ".weights.h5"
            else:
                return save_dir + ".weights.h5"
        else:
            if save_dir.endswith(".keras"):
                return save_dir
            else:
                return save_dir + ".keras"
