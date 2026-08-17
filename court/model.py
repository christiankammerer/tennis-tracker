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

    @property
    def left_double_near(self):
        return np.array([
        [0,0], 
        [self.singles_sideline, 0], 
        [0, self.court_l/2],
        [self.singles_sideline, self.court_l/2],
        ])
    
    @property
    def right_double_near(self):
        return np.array([
        [self.court_w - self.singles_sideline, 0], 
        [self.court_w, 0], 
        [self.court_w - self.singles_sideline, self.court_l/2],
        [self.court_w, self.court_l/2],
        ])

    @property
    def left_double_far(self):
        return np.array([
        [0, self.court_l], 
        [self.singles_sideline, self.court_l], 
        [0, self.court_l/2],
        [self.singles_sideline, self.court_l/2],
        ])

    @property
    def right_double_far(self):
        return np.array([
        [self.court_w - self.singles_sideline, self.court_l], 
        [self.court_w, self.court_l], 
        [self.court_w - self.singles_sideline, self.court_l/2],
        [self.court_w, self.court_l/2],
        ])

court = Court()
print(court.right_double_far)
