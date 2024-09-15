import pandas as pd
import os


def save_df(dataframe, path, filename, header=None):
    """
    Saves a DataFrame to a CSV file, creating the directory if it doesn't exist.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to save.
    path (str): The directory path where the file will be saved.
    filename (str): The name of the CSV file.
    header (list of str, optional): Custom header for the CSV file. Defaults to None.

    Example:
    >>> save_df(df, "./out/experiment", "results.csv", header=["Column1", "Column2"])
    """
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe.to_csv(path + '/' + filename, index=False, header=header)