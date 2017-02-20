import os
import sys


def make_dir_if_not_exists(path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return path


def main_script_name():
    return os.path.basename(sys.argv[0]).split('.')[0]


def make_symlink(target, link):
    make_dir_if_not_exists(link)
    if os.path.exists(link) and os.path.islink(link):
        os.unlink(link)
    if not os.path.exists(link):
        cmd = "ln -s %s %s" % (target, link)
        os.system(cmd)


def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
