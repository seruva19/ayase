#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import numpy as np
from .network import IDreveal


class ComputeIdReveal:
    def __init__(self, time, device, weights_file):
        self.time = time
        self.net = IDreveal(time=self.time, device=device, weights_file=weights_file)

    def __call__(self, feats):
        return self.net(np.stack(feats, 0))[0]
