from urllib import request
import os
import pandas as pd

"""
Table Access Protocol (TAP)

This is the link to the NEA table of confirmed planets (from "ps" == Planetary Systems) 
with default flag = 1 (default_flag+=+1)
The table is saved in CSV format (format=csv)
"""

download_link = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps+where+default_flag+=+1&format=csv'

def _create_data_dir():
    """ Create empty directory where nea_full.csv will be stored """
    current_dir = os.getcwd()
    table_directory = os.path.join(current_dir, 'nea_tables')
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)


def _check_data_dir():
    current_dir = os.getcwd()
    table_directory = os.path.join(current_dir, 'nea_tables')
    return os.path.exists(table_directory)


def get_data_dir():
    """ Return directory where nea_full.csv is stored """
    if not _check_data_dir():
        _create_data_dir()
        
    current_dir = os.getcwd()
    return os.path.join(current_dir, 'nea_tables')


def download_nea_table():
    """ Download and save nea_full.csv table in ./nea_tables directory  """
    local_file = os.path.join(get_data_dir(), 'nea_full.csv')

    with request.urlopen(download_link) as response:
        data = response.read()

    with open(local_file, 'wb') as f:
        f.write(data)

def open_nea_table():
    """ Opens the NEA full table as a pandas dataframe"""
    current_dir = os.getcwd()
    table_directory = os.path.join(current_dir, 'nea_tables', 'nea_full.csv')
    nea_full_table = pd.read_csv(table_directory)

    return nea_full_table