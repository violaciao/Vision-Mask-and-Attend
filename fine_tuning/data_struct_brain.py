import os
import re
import pickle
import scipy.misc
import numpy as np
import matplotlib.cm as cm


def get_indicators(pickle_path):
    indicator_list = []
    for i in os.listdir(pickle_path):
        i = i.split("_") 
        ind_dir = "_".join([i[1], i[2]])
        indicator_list.append(ind_dir)
    return set(indicator_list)

def j(a, b):
    return os.path.join(a, b)


def construct_dataset(set_folders=True):

	DATA_PATH = ".../data/data_brain"
	PICKLE_PATH = ".../data/data_brain/pickle_files"

	indicators = get_indicators(PICKLE_PATH)


	if set_folders:  # set as "True" if first time run

	    for indicator in indicators:
	        indicator_path = os.path.join(DATA_PATH, indicator)
	        os.makedirs(indicator_path)

	        for sub in ['train', 'val', 'test']:
	            sub_path = os.path.join(indicator_path, sub)
	            os.makedirs(sub_path)

	            for cat in [0, 1, 2]:
	                cat_path = os.path.join(sub_path, str(cat))
	                os.makedirs(cat_path)	


	for indicator in indicators:
    
	    for sub in ['train', 'validation', 'testing']:

	        for item in os.listdir(PICKLE_PATH):
	            
	            i = item.split("_")
	            ind_dir = "_".join([i[1], i[2]])
	            sub_dir = i[0]
	            
	            pickle_dir = os.path.join(PICKLE_PATH, item)
	            
	            if (ind_dir == indicator) and (sub_dir == sub):
	                
	                if i[3] != 'all': 
	                    tag = i[3]
	                else:
	                    tag = i[4]
	                    
	                if tag == 'x':
	                    with open(pickle_dir, 'rb') as f:
	                        data = pickle.load(f)
	                elif tag == 'y':
	                    with open(pickle_dir, 'rb') as f:
	                        label = pickle.load(f)
	                else:
	                    with open(pickle_dir, 'rb') as f:
	                        ID = pickle.load(f)
	        
	        for i, i_label in enumerate(label):
	            
	            if sub == 'validation':
	                sub = 'val'
	            elif sub == 'testing':
	                sub = 'test'
	                
	            cat = str(i_label.argmax())
	            
	            dst_path = j( j( j(DATA_PATH, indicator), sub), cat)
	            dst_dir = j(dst_path, ID[i]+'_'+str(i).zfill(4)+'.png')
	            
	            scipy.misc.imsave(dst_dir, data[i].reshape((256, 256)))


if __name__ == '__main__':
	construct_dataset()
