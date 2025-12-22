"""
Wavira: Wi-Fi-based Person Re-Identification

A deep learning framework for person re-identification using Wi-Fi
Channel State Information (CSI), based on the WhoFi architecture.
"""

__version__ = "0.1.0"

from wavira.models.whofi import WhoFi
from wavira.data.dataset import CSIDataset
from wavira.losses.inbatch_loss import InBatchNegativeLoss

__all__ = ["WhoFi", "CSIDataset", "InBatchNegativeLoss"]
