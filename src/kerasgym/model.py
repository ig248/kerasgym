from abc import ABCMeta, abstractmethod
from keras.models import load_model
import simplejson as json


def history_join(old_history, new_history):
    """Join two history dicts"""
    if not old_history:
        return new_history
    if not new_history:
        return old_history
    history = old_history
    for key in history:
        history[key].extend(new_history[key])
    return history


class GymModel:
    """Base class for building persistent kerasgym models"""
    __metaclass__ = ABCMeta

    def __init__(self, filename=None):
        self._model = None
        self._history = None
        self._epoch = 0

    @property
    def is_initialized(self):
        return False if self._model is None else True

    @property
    def epoch(self):
        return self._epoch

    def init_model(self):
        """Initialize keras model per user spec"""
        self._model = self.model()

    def load_model(self, model_file):
        """Load complete model from file"""
        self._model = load_model(model_file)

    def load_history(self, history_file):
        """Load history from json file"""
        with open(history_file) as hf:
            self._history = json.load(hf)
        self._epoch = len(self._history['loss'])

    def save_model(self, model_file):
        """Save current model to file"""
        self._model.save(model_file)

    def save_history(self, history_file):
        """Save current history to json file"""
        with open(history_file, 'w') as hf:
            json.dump(self._history, hf, indent=2)

    def summary(self):
        if self._model:
            self._model.summary()

    def train_update(self, epochs=10):
        if self._model is None:
            return
        k_history = self.train(self._model, epochs=epochs, initial_epoch=self._epoch)
        new_history = k_history.history  # train returns a keras History object
        self._history = history_join(self._history, new_history)

    # These functions must be implemented:

    @staticmethod
    @abstractmethod
    def model(self):
        """Return a ready-to-use compiled keras model"""
        model = None
        return model

    @staticmethod
    @abstractmethod
    def train(self, model, epochs=10, initial_epoch=0):
        """Perform specified number of epochs, and return history object"""
        new_history = None
        return new_history
