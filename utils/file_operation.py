import pandas as pd
import os


def save_df(dataframe, path, filename, header=None):
    """
    Saving the dataframe and creating the path if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe.to_csv(path + '/' + filename, index=False, header=header)