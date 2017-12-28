import pickle
import os

from preprocess.process_leaf import process_scan_leaf

def main():
    leafscan_dict = pickle.load(open('./train.pickle', 'rb'))
    all_classes = leafscan_dict.keys()

    if os.path.exists('./process_train') \
        and len(os.listdir('./process_train')) == 12605:
        print('Handling has been done.')
        return

    # Example for showing the orignal pic
    for family in all_classes[14:]:
        for im_id in leafscan_dict.get(family):
            ex_path = os.path.join('./train/', im_id + '.jpg')
            process_scan_leaf(ex_path, output_folder='./process_train/')

if __name__ == '__main__':
    main()
