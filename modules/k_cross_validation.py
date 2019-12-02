
def split(_, features, labels, test_size=0.3):
    return _.train_test_split(features, labels, test_size = test_size)