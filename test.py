from tempfile import TemporaryDirectory
import os
import os.path as osp
import numpy as np
from PIL import Image

from dataloader import *


if __name__ == '__main__':
    test = MicrogramsDataset()
    test.create_train_set()