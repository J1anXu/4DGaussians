from enum import Enum

class ImpScoreType(Enum):
    OPACITY = "opacity"
    OPACITY_MOVING_PATH = "opacity_and_movingInfo"
    INIT_BLENDING_WEIGHT = "init_blending_weight"
    ALL_TIME_BLENDING_WEIGHT = "all_time_blending_weight"


class ZPruneType(Enum):
    OPACITY_SORT = "opacity_sort"
    IMP_SCORE = "imp_score"