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
            if not LeafScan_dict.get(img_info['Family']):
                LeafScan_dict[img_info['Family']] = [img_id]
            else:
                LeafScan_dict[img_info['Family']].append(img_id)

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
