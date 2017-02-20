def _linear(progress, l, r):
    assert 0. <= progress <= 1.
    return l + (r - l) * progress


def linear_1_0(progress):
    return _linear(progress, 1, 0)


def linear_05_00(progress):
    return _linear(progress, .5, .0)
