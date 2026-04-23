"""
CSV I/O utilities for loading data into NumPy arrays.
"""

import numpy as np


def load_csv(filepath, delimiter=',', skip_header=True, missing_values='', 
             filling_values=np.nan):
    """
    Load a CSV file into a NumPy array.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    delimiter : str, default=','
        Character that separates values (comma, tab, etc.)
    skip_header : bool, default=True
        Whether to skip the first row (column names)
    missing_values : str or list, default=''
        What represents missing data in the file
    filling_values : float, default=np.nan
        What to replace missing values with
        
    Returns
    -------
    data : ndarray
        2D NumPy array with the loaded data
        
    Examples
    --------
    >>> data = load_csv('students.csv')
    >>> data.shape
    (100, 5)  # 100 students, 5 features
    
    Notes
    -----
    - Uses np.genfromtxt for robust CSV parsing
    - All data is converted to float64
    - Empty strings are treated as missing by default
    
    Time Complexity
    ---------------
    O(n*m) where n=rows, m=columns
    
    Space Complexity
    ----------------
    O(n*m) for storing the output array
    """
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=1 if skip_header else 0,
        missing_values=missing_values,
        filling_values=filling_values,
        dtype=np.float64
    )
    
    return data