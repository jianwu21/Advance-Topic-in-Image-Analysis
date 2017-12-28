from os import listdir, path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle
import shutil


def generate_data(learning_type='train'):
    '''
    `learning_type` must be one of `test` or `train`
    '''
    if learning_type not in ['train', 'test']:
        raise ValueError(
            '\'learning_type\' can be only one of [\'train\', \'test\']')

    dirpath = path.dirname(path.abspath(__file__))

    if path.exists(path.join(dirpath, learning_type + '.pickle')):
        print('File {} exists!'.format(learning_type + '.pickle'))
        return

    current_file_list = listdir(dirpath)

    files_list = listdir(path.join(dirpath, learning_type))

    all_img_ids = set([
        im_file.split('.')[0]
        for im_file in files_list
    ])

    LeafScan_dict = {}

    for img_id in all_img_ids:
        img_info = _parse_xml(
            path.join(dirpath, learning_type, img_id+'.xml')
        )

        if img_info['Content'] == 'LeafScan':
            LeafScan_dict[img_id] = img_info['Family']

    pickle.dump(
        LeafScan_dict,
        open(path.join(dirpath, learning_type + '.pickle'), 'wb')
    )

    return


def _parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pared_file = {}

    for field in root:
        pared_file.update(
            {field.tag: field.text}
        )

    return pared_file


if __name__ == '__main__':
    generate_data()
    generate_data(learning_type='test')

    leafscan_dict = pickle.load(open('./train.pickle', 'rb'))
    all_classes = list(set(leafscan_dict.values()))
    label_dict = dict(zip(all_classes, range(len(all_classes))))
    print(label_dict)

    pickle.dump(
        label_dict,
        open('./label_dict.pickle', 'wb')
    )
