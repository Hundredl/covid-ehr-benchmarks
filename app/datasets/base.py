import pickle


def load_data(dataset_type):
    # Load data
    data_path = f"datasets/{dataset_type}/processed_data/"
    x = pickle.load(open(data_path + "x.pkl", "rb"))
    y = pickle.load(open(data_path + "y.pkl", "rb"))
    x_lab_length = pickle.load(open(data_path + "visits_length.pkl", "rb"))

    return x, y, x_lab_length


def load_data_split(dataset_type):
    data_path = f"datasets/{dataset_type}/processed_data/"
    x_train = pickle.load(open(data_path + "train_xs.pkl", "rb"))
    y_train = pickle.load(open(data_path + "train_ys.pkl", "rb"))
    x_train_lab_length = pickle.load(open(data_path + "train_lens.pkl", "rb"))
    x_valid = pickle.load(open(data_path + "valid_xs.pkl", "rb"))
    y_valid = pickle.load(open(data_path + "valid_ys.pkl", "rb"))
    x_valid_lab_length = pickle.load(open(data_path + "valid_lens.pkl", "rb"))
    x_test = pickle.load(open(data_path + "test_xs.pkl", "rb"))
    y_test = pickle.load(open(data_path + "test_ys.pkl", "rb"))
    x_test_lab_length = pickle.load(open(data_path + "test_lens.pkl", "rb"))

    return x_train, y_train, x_train_lab_length, x_valid, y_valid, x_valid_lab_length, x_test, y_test, x_test_lab_length
