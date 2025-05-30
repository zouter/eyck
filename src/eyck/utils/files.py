import os
import pathlib

def get_git_root(cwd=None):
    """
    Gets the first parent root with a a .git folder
    """
    if cwd is None:
        cwd = os.getcwd()
    # go back until we find the git directory, signifying project root
    while ".git" not in os.listdir(cwd) and os.path.realpath(cwd) != "/":
        cwd = os.path.dirname(cwd)

    return pathlib.Path(cwd)


def get_results():
    return get_git_root() / "results"

def get_output():
    return get_git_root() / "output"


def get_code():
    return get_git_root() / "code"

def get_data():
    return get_git_root() / "data"

def get_software():
    return get_git_root() / "software"
