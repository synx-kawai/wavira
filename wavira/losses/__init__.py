"""Loss functions for person re-identification training."""

from wavira.losses.inbatch_loss import InBatchNegativeLoss, TripletLoss

__all__ = ["InBatchNegativeLoss", "TripletLoss"]
