"""Module for early stopping class
"""
import numpy as np


class EarlyStopper:
    """Early stopper check."""

    def __init__(self, patience=3, min_abs_delta=1e-2, min_rel_delta=0.0):
        """EarlyStopper. Use to stop if the validation loss starts increasing.
        The validation loss is increasing if

            L > Lmin + abs_delta + rel_delta * |Lmin|,

        where `L` is the current validation loss, `Lmin` is the minimum validation loss found so
        far, and `abs_delta` and `rel_delta` are absolute and relative tolerances to the increase,
        respectively.


        Args:
            patience (int, optional): How many consecutive iterations to wait before stopping.
                Defaults to 3.
            min_abs_delta (float, optional): Absolute tolerance to the increase. Defaults to 1e-2.
            min_rel_delta (float, optional): Relative tolerance to the increase. Defaults to 0.0.
        """
        self.patience = patience
        self.min_abs_delta = min_abs_delta
        self.min_rel_delta = min_rel_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Check whether it is time to stop, and update the internal of EarlyStopper.

        Args:
            validation_loss (float): Current validation loss

        Returns:
            stop (boolean): Whether it is time to stop (True) or not (False).
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False

        if self.min_validation_loss is np.inf:
            return False

        loss_threshold = (
            self.min_validation_loss
            + self.min_abs_delta
            + self.min_rel_delta * np.abs(self.min_validation_loss)
        )

        if validation_loss > loss_threshold:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
