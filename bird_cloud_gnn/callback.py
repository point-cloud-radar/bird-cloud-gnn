import numpy as np
from torch.utils.tensorboard import SummaryWriter
from bird_cloud_gnn.early_stopper import EarlyStopper


class TensorboardCallback:
    """Callback to populate Tensorboard.

    This class provides a callback function to populate Tensorboard with scalar and histogram summaries
    during training. The callback function takes in epoch values and adds scalar and histogram summaries
    to Tensorboard for each field in the epoch values that matches certain criteria.

    Args:
        **kwargs: Additional arguments to pass to the SummaryWriter constructor.

    Attributes:
        writer: A SummaryWriter object used to write summaries to Tensorboard.
    """

    def __init__(self, **kwargs):
        self.writer = SummaryWriter(**kwargs)

    def __call__(self, epoch_values):
        """Callback function to populate Tensorboard with scalar and histogram summaries.

        Args:
            epoch_values: A dictionary containing the values for each field at the current epoch.

        Returns:
            False, indicating that the training should continue.
        """
        epoch = epoch_values["epoch"]
        layer_names = [
            key
            for key in epoch_values.keys()
            if "Loss/" in key or "Rate" in key or "Accuracy" in key
        ]
        for field in layer_names:
            self.writer.add_scalar(field, epoch_values[field], epoch)

        layer_names = [key for key in epoch_values.keys() if "Layer/" in key]
        for field in layer_names:
            self.writer.add_histogram(field, epoch_values[field].numpy(), epoch)
            self.writer.add_scalar(
                field.replace("Layer", "LayerAverage"),
                np.average(epoch_values[field].numpy()),
                epoch,
            )

        return False


class EarlyStopperCallback:
    """
    Callback to check early stopping.

    This callback is used to check if the training should be stopped early based on the validation loss.
    """

    def __init__(self, **kwargs):
        """Input arguments are passed to EarlyStopper."""
        self.early_stopper = EarlyStopper(**kwargs)

    def __call__(self, epoch_values):
        return self.early_stopper.early_stop(epoch_values["Loss/test"])


class CombinedCallback:
    """Helper to combine multiple callbacks.

    This class allows multiple callbacks to be combined into a single callback. The callbacks are called in the given
    sequence and if one of them returns True, the subsequent callbacks are not called.

    Args:
        callbacks (iterable): List of callbacks to be combined.

    Returns:
        bool: True if any of the callbacks return True, False otherwise.
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, epoch_values):
        return_value = False
        for callback in self.callbacks:
            return_value = return_value or callback(epoch_values)
        return return_value
