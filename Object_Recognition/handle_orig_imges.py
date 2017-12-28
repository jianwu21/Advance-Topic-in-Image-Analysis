import pickle
import os

from preprocess.process_leaf import process_scan_leaf

def process_train():
    leafscan_dict = pickle.load(open('./train.pickle', 'rb'))
    all_img_ids = leafscan_dict.keys()

    if os.path.exists('./process_train') \
        and len(os.listdir('./process_train')) == 12605:
        print('Handling has been done.')
        return

    # Example for showing the orignal pic
    for im_id in all_img_ids:
        ex_path = os.path.join('./train/', im_id + '.jpg')
        process_scan_leaf(ex_path, output_folder='./process_train/')


def process_test():
    leafscan_dict = pickle.load(open('./test.pickle', 'rb'))
    all_img_ids = leafscan_dict.keys()

    if os.path.exists('./process_test') \
        and len(os.listdir('./process_test')) == 12605:
        print('Handling has been done.')
        return

    # Example for showing the orignal pic
    for im_id in all_img_ids:
        ex_path = os.path.join('./test/', im_id + '.jpg')
        process_scan_leaf(ex_path, output_folder='./process_test/')

if __name__ == '__main__':
    process_train()
    process_test()
