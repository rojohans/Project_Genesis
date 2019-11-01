import os

def Path():
    #
    # This function is used to get the path of the project directory. This is useful when files are to be saved/loaded.
    #
    return os.path.dirname(__file__)