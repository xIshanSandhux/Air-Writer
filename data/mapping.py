import scipy.io

def get_emnist_mapping():
    """
    Returns a dictionary mapping labels (0-25) to lowercase letters 'a' to 'z'.
    EMNIST Letters dataset has labels 1-26, so we subtract 1 during training.
    """
    return {i: chr(97 + i) for i in range(26)}  # 0 → 'a', 1 → 'b', ..., 25 → 'z'


