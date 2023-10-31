"""Module for early stopping class
"""
import numpy as np


class EarlyStopper:
    """Early stopper check.

    This class is used to stop the training process if the validation loss starts increasing. The validation loss is
    considered to be increasing if it is greater than the minimum validation loss found so far plus an absolute and/or
    relative tolerance. The class keeps track of the minimum validation loss found so far and the number of consecutive
    iterations where the validation loss has increased. If the number of consecutive iterations where the validation
    loss has increased exceeds a certain threshold, the training process is stopped.

    Attributes:
        patience (int): How many consecutive iterations to wait before stopping.
        min_abs_delta (float): Absolute tolerance to the increase.
        min_rel_delta (float): Relative tolerance to the increase.
        counter (int): Number of consecutive iterations where the validation loss has increased.
        min_validation_loss (float): Minimum validation loss found so far.

    Methods:
        __init__(self, patience=3, min_abs_delta=1e-2, min_rel_delta=0.0): Initializes the EarlyStopper object.
        early_stop(self, validation_loss): Checks whether it is time to stop, and updates the internal state of the
            EarlyStopper object.

    Example usage:
        early_stopper = EarlyStopper(patience=5, min_abs_delta=0.1, min_rel_delta=0.01)
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader)
            val_loss = validate(model, val_loader)
            if early_stopper.early_stop(val_loss):
                print(f"Validation loss has been increasing for {early_stopper.patience} consecutive epochs. "
                      f"Training stopped.")
                break
    """

    def __init__(self, patience=3, min_abs_delta=1e-2, min_rel_delta=0.0):
        """Initializes the EarlyStopper object.

        Args:
            patience (int, optional): How many consecutive iterations to wait before stopping. Defaults to 3.
            min_abs_delta (float, optional): Absolute tolerance to the increase. Defaults to 1e-2.
            min_rel_delta (float, optional): Relative tolerance to the increase. Defaults to 0.0.
        """
        self.patience = patience
        self.min_abs_delta = min_abs_delta
        self.min_rel_delta = min_rel_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Checks whether it is time to stop, and updates the internal state of the EarlyStopper object.

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
