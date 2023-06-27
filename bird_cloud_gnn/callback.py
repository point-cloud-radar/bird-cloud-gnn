from torch.utils.tensorboard import SummaryWriter
from bird_cloud_gnn.early_stopper import EarlyStopper


class TensorboardCallback:
    """Callback to populate Tensorboard"""

    def __init__(self):
        self.writer = SummaryWriter()

    def __call__(self, epoch_values):
        epoch = epoch_values["epoch"]
        for field in ["Loss/train", "Loss/test", "Accuracy/train", "Accuracy/test"]:
            self.writer.add_scalar(field, epoch_values[field], epoch)
        return False


class EarlyStopperCallback:
    """Callback to check early stopping."""

    def __init__(self, **kwargs):
        """Input arguments are passed to EarlyStopper."""
        self.early_stopper = EarlyStopper(**kwargs)

    def __call__(self, epoch_values):
        return self.early_stopper.early_stop(epoch_values["Loss/test"])


class CombinedCallback:
    """Helper to combine multiple callbacks."""

    def __init__(self, callbacks):
        """
        Args:
            callbacks (iterable): List of callbacks. These are called in the given sequence and
                if one of them returns True, the subsequents are not called.
        """
        self.callbacks = callbacks

    def __call__(self, epoch_values):
        return_value = False
        for callback in self.callbacks:
            return_value = return_value or callback(epoch_values)
        return return_value
