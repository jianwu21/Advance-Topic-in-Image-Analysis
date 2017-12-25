import cv2
from os import listdir, path
import sqlite3 as sqlite
import xml.etree.ElementTree as ET

def main():
    dirpath = path.dirname(path.abspath('__file__'))
    files_list = listdir(path.join(dirpath, 'train/'))

    all_img = set([
        im_file.split('.')[0],
        for im_file in files_list
    ])


    # build one connection to db file
    con = sqlite.connect('train.db')
    c = con.cursor()
    print('creating table...')
    c.execute(
        '''
        create table img(
            id INTEGER PRIMARY KEY AUTOINCREMENT not null,
            rgb,
            label,
        );
        '''
    )
    print('table has been created')
