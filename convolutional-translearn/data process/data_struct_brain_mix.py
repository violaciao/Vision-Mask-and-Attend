import os
import re
import pickle
import random
import shutil
import scipy.misc
import numpy as np



def j(a, b):
	return os.path.join(a, b)


def construct_dataset(set_folders=True):

	FROM_PATH = "../data/data_brain_no2"
	TO_PATH = "../data/data_brain_mix"

	random.seed(1)

	# e.g., FROM_PATH/T1_all/train/0/TCGA-06-0154_i.png
	#		TO_PATH/train/0/TCGA-06-0154+T1_all+i.png


	if set_folders:  # set as "True" if first time run

		n = 0

		for i in ['train', 'val', 'test']:
			sub_folder = j(TO_PATH, i)
			os.makedirs(sub_folder)

			for ii in ['0', '1']:
				os.makedirs( j(sub_folder, ii) )


		for indicator in os.listdir(FROM_PATH):
			indicator_path = j(FROM_PATH, indicator)

			for sub in os.listdir(indicator_path):
				sub_path = j(indicator_path, sub)

				for cat in os.listdir(sub_path):
					cat_path = j(sub_path, cat)

					for img in os.listdir(cat_path):
						# TO_PATH/train/0/TCGA-06-0154+T1_all+i.png
						name = img.split('_')[0] + '+' + indicator + '+' + str(n) + '.png'
						file_path = j( j(TO_PATH, sub), cat)

						src = j(cat_path, img)
						dst = j(file_path, name)

						shutil.copy(src, dst)
						n += 1
			



if __name__ == '__main__':
	construct_dataset()
