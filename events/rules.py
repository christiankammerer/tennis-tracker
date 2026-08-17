from __future__ import annotations

from typing import Literal, Tuple

from court import Court

PlayType = Literal["serve", "rally"]
Mode = Literal["singles", "doubles"]
Side = Literal["near", "far"]
Region = Literal[
    "serve_near_deuce",
    "serve_far_deuce",
    "serve_near_ad",
    "serve_far_ad",
    "back_near",
    "back_far",
    "left_double_near",
    "right_double_near",
    "left_double_far",
    "right_double_far"
]

# Where did the ball bounce? 

def get_region(x: float, y: float) -> Region:
    
