KerasGym
========

This package was written to simplify the task of keeping track of
``keras`` deep learning models while working on a real-world problem. I
found that being able to re-visit previous experiments, especially for
comparing training curves and using saved models for prediction, was
rather useful.

This gym is under construction.

Quick start
-----------

Requires ``keras``.

Installation:

::

    $ git clone https://github.com/ig248/kerasgym
    $ cd kerasgym

List command-line options:

::

    $ kerasgym --help

Run 10 epochs:

::

    $ kerasgym --verbose --path examples --model toymodel --epochs 10

Continue training from last save point:

::

    $ kerasgym --verbose --path examples --model toymodel --continue --epochs 10

Saved files
-----------

Two files are saved for each model: - ``toymodel.h5``: a complete state
of the ``keras`` model, including weights, for future use -
``toymodelhistory.json``: a plain-text json file containing combined
history of all metrics over training epochs

See https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
and https://keras.io/callbacks/#history for further information.

Building your own models
------------------------

``kerasgym`` currently looks for a file ``path/model.py`` defining a
``keras`` model with the following structure:

.. code:: python

    from kerasgym import GymModel

    class Model(GymModel):

        def model(self):
            """Return a ready-to-use compiled keras model"""
            model = Sequential()
            ...
            optimizer = Adam(lr=0.0001)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

            return model

        def train(self, model, epochs, initial_epoch):
            """Perform specified number of epochs, and return history object"""
            history = model.fit(epochs=epochs + initial_epoch,
                                initial_epoch=initial_epoch,
                                ...)
            return history

It is expected to contain a class ``Model``, inheriting from
``kerasgym.GymModel``, and must implement the static methods ``model``
and ``train``. See ``examples`` for a sample implementation.

Credits
-------

-  ```keras`` <https://keras.io/>`__
-  ```Distribute`` <http://pypi.python.org/pypi/distribute>`__
-  ```Buildout`` <http://www.buildout.org/>`__
-  ```modern-package-template`` <http://pypi.python.org/pypi/modern-package-template>`__
