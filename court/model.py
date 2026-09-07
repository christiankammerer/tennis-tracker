court_w = 10.97
court_l = 23.77
singles_sideline = 1.37
sl_to_net = 6.40

import numpy as np
from dataclasses import dataclass


@dataclass
class Court:
    court_w: float = court_w
    court_l: float = court_l
    singles_sideline: float = singles_sideline
    sl_to_net: float = sl_to_net    

    @property
    def court_corners(self):
        return np.array([
            [0, self.court_l],      # near-left
            [self.court_w, self.court_l],   # near-right
            [self.court_w, 0],       # far-right
            [0, 0],          # far-left
        ])

    def _alley_quad(self, x0: float, x1: float, y_near: float, y_far: float) -> np.ndarray:
        """Return a non-crossing quad: near-left, near-right, far-right, far-left.

        Near is y = court_l (camera end); far is y = 0. Vertices must wind in
        that order so contourArea / pointPolygonTest see a real rectangle.
        """
        return np.array(
            [
                [x0, y_near],
                [x1, y_near],
                [x1, y_far],
                [x0, y_far],
            ]
        )

    @property
    def left_double_near(self):
        return self._alley_quad(
            0.0, self.singles_sideline, self.court_l, self.court_l / 2
        )

    @property
    def right_double_near(self):
        return self._alley_quad(
            self.court_w - self.singles_sideline,
            self.court_w,
            self.court_l,
            self.court_l / 2,
        )

    @property
    def left_double_far(self):
        return self._alley_quad(
            0.0, self.singles_sideline, self.court_l / 2, 0.0
        )

    @property
    def right_double_far(self):
        return self._alley_quad(
            self.court_w - self.singles_sideline,
            self.court_w,
            self.court_l / 2,
            0.0,
        )
