from libraries import *


def dataset_info(df):

    print(df.shape)
    print(df.columns)
    print(df.describe().T)
