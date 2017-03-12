import os
from common.os_tools import make_dir_if_not_exists


class ExperimentData(object):
    def __init__(self, base_path='.'):
        self.base_path = base_path
        make_dir_if_not_exists(self.base_path)

    def path(self, p):
        base_path = os.path.abspath(self.base_path)
        p = os.path.join(base_path, p)
        make_dir_if_not_exists(p)
        return p
