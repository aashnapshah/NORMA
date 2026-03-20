from .eicu import EICUDataset
from .chs import CHSDataset

DATASETS = {
    "eicu": EICUDataset,
    "chs": CHSDataset,
}
