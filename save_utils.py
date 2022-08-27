import pickle


def save_pickle(save_file, save_path):
    '''Function for saving pickle file'''
    with open(save_path, 'wb') as f:
        pickle.dump(save_file, f)

    print(f'Saved at {save_path}')


def load_pickle(save_path):
    '''Function for loading pickle file'''
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded at {save_path}')
    return data
