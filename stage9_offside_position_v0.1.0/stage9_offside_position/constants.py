from __future__ import annotations

LEGAL_LANDMARKS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
LEGAL_LANDMARK_SET = frozenset(LEGAL_LANDMARKS)
EXCLUDED_ARM_LANDMARKS = frozenset({"left_elbow", "right_elbow", "left_wrist", "right_wrist"})

DEFAULT_EPSILON_M = 1e-9
DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0

LABEL_OFFSIDE = "OFFSIDE_POSITION"
LABEL_ONSIDE = "ONSIDE"
LABEL_TOUCHER = "TOUCHER_EXCLUDED"
LABEL_UNAVAILABLE = "UNAVAILABLE"
