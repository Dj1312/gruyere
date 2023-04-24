from enum import IntEnum


class TouchState(IntEnum):
    EMPTY = -1
    INVALID = 0
    IMPOSSIBLE = 0
    EXISTING = 1
    VALID = 2
    FREE = 3
    RESOLVING = 4
    TEST = 100


class PixelState(IntEnum):
    IMPOSSIBLE = 0
    EXISTING = 1
    POSSIBLE = 2
    REQUIRED = 3
    TEST = 100


class DesignState(IntEnum):
    VOID = -1
    UNASSIGNED = 0
    SOLID = 1
    TEST = 100


class FreeState(IntEnum):
    VOID = -1
    SOLID = 1
    TEST = 100
