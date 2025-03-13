import os
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl

TEST_CASES = {
    # 14: {
    #     'name': 'BreastCancer',
    #     'n_cluster': 2,
    #     'test_points': ['30-39', 'premeno', '30-34', '0-2', 'no', 3, 'left', 'left_low', 'no']
    # },
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    # 80: {
    #     'name': 'Digits',
    #     'n_cluster': 10,
    #     'test_points': [0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14, 7, 1, 0, 0]
    # },
    109: {
        'name': 'Wine',
        'n_cluster': 3,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    236: {
        'name': 'Seeds',
        'n_cluster': 3,
        'test_points': [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])

def load_dataset(data: dict, file_csv: str = '', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    # label_name = data['data']['target_col']
    print('DATASET UCI id=', data['data']['uci_id'], data['data']['name'], f"{data['data']['num_instances']} x {data['data']['num_features']}")  # Mã + Tên bộ dữ liệu
    # print('data abstract=', data['data']['abstract'])  # Tóm tắt bộ dữ liệu
    # print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    # print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    # print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    # colnames = ['Area', 'Perimeter']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    # print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    # Trích xuất ma trận đặc trưng X (loại trừ nhãn lớp)
    return {'data': data['data'], 'ALL': df.iloc[:, :].values, 'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1:].values}


# Lấy dữ liệu từ ổ cứng
def fetch_data_from_local(name_or_id=53, folder: str = './dataset', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    if isinstance(name_or_id, str):
        name = name_or_id
    else:
        name = TEST_CASES[name_or_id]['name']
    _folder = os.path.join(folder, name)
    fileio = os.path.join(_folder, 'api.json')
    if not os.path.isfile(fileio):
        print(f'File {fileio} not found!')
    with open(fileio, 'r') as cr:
        response = cr.read()
    return load_dataset(json.loads(response),
                        file_csv=os.path.join(_folder, 'data.csv'),
                        header=header, index_col=index_col, usecols=usecols, nrows=nrows)


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53, header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        _rs = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        response = _rs.read()
        _rs.close()
        return load_dataset(json.loads(response),
                            header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')
