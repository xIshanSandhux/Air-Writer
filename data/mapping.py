import scipy.io

def get_emnist_mapping():
    mat = scipy.io.loadmat("data/emnist-byclass.mat")
    dataset = mat["dataset"]

    # Access the mapping matrix at index 2
    mapping = dataset[0][0][2]

    # Build a dictionary: {label_index: character}
    label_map = {int(label): chr(int(code)) for label, code in mapping}
    return label_map


