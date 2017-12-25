from os import listdir, path, remove
import sqlite3 as sqlite
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle


def generate_train_data():
    dirpath = path.dirname(path.abspath(__file__))
    current_file_list = listdir(dirpath)

    files_list = listdir(path.join(dirpath, 'train/'))

    all_img_ids = set([
        im_file.split('.')[0]
        for im_file in files_list
    ])


    # build one connection to db file
    con = sqlite.connect('imgs.db')
    c = con.cursor()
    c.execute(
        'drop table if exists train_img'
    )
    print('creating table...')
    c.execute(
        '''
        create table train_img(
            id INTEGER PRIMARY KEY not null,
            img,
            class_id INTEGER,
            content STRING,
            family STRING,
            genus STRING,
            species STRING,
            author STRING
        );
        '''
    )
    print('table {} has been created'.format('train_img'))

    for img_id in all_img_ids:
        img_info = _parse_xml(
            './train/' + img_id + '.xml'
        )
        img_rgb = plt.imread('./train/' + img_id + '.jpg').astype('float')

        if img_info['Content'] != 'Flower':
            # print('Img {} is not belong to \'Flower\''.format(img_id))
            continue

        c.execute(
            '''
            insert into train_img
            (id, img, class_id, content, family, genus, species, author)
            values
            (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                int(img_info['MediaId']),
                pickle.dumps(img_rgb),
                int(img_info['ClassId']),
                img_info['Content'],
                img_info['Family'],
                img_info['Genus'],
                img_info['Species'],
                img_info['Author']
            )
        )
        print('Inserting the image: {}'.format(img_id))
    con.commit()
    con.close()


def generate_test_data():
    dirpath = path.dirname(path.abspath(__file__))
    current_file_list = listdir(dirpath)

    files_list = listdir(path.join(dirpath, 'test/'))

    all_img_ids = set([
        im_file.split('.')[0]
        for im_file in files_list
    ])


    # build one connection to db file
    con = sqlite.connect('imgs.db')
    c = con.cursor()
    c.execute(
        'drop table if exists test_img'
    )
    print('creating table...')
    c.execute(
        '''
        create table test_img(
            id INTEGER PRIMARY KEY not null,
            img,
            class_id INTEGER,
            content STRING,
            family STRING,
            genus STRING,
            species STRING,
            author STRING
        );
        '''
    )
    print('table {} has been created'.format('test_img'))

    for img_id in all_img_ids:
        img_info = _parse_xml(
            './test/' + img_id + '.xml'
        )
        img_rgb = plt.imread('./test/' + img_id + '.jpg').astype('float')

        if img_info['Content'] is not 'Flower':
            # print('Img {} is not belong to \'Flower\''.format(img_id))
            continue

        c.execute(
            '''
            insert into test_img
            (id, img, class_id, content, family, genus, species, author)
            values
            (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                int(img_info['MediaId']),
                pickle.dumps(img_rgb),
                int(img_info['ClassId']),
                img_info['Content'],
                img_info['Family'],
                img_info['Genus'],
                img_info['Species'],
                img_info['Author']
            )
        )
        print('Inserting the image: {}'.format(img_id))
    con.commit()
    con.close()


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
    generate_train_data()
    generate_test_data()
