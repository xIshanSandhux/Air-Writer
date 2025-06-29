import scipy.io

mat = scipy.io.loadmat("data/emnist-byclass.mat")
print(mat.keys())
print(type(mat['dataset']))
print(mat['dataset'].dtype)