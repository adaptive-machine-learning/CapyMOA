from capymoa._pickle import JPickler, JUnpickler


def save_model(model, filename):
    """Save a model to a file.
    param model: The model to save.
    param filename: The file to save the model to."""

    with open(filename, "wb") as fd:
        JPickler(fd).dump(model)


def load_model(filename):
    """Load a model from a file.
    param filename: The file to load the model from."""

    with open(filename, "rb") as fd:
        return JUnpickler(fd).load()
