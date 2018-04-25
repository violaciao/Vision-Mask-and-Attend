import os
import random
import shutil
from math import floor


def main():
	random.seed(1)
	# split = 0.8

	LUAD_Patches_FROM =  '/scratch/nsr3/TCGA-LUAD/TCGA-LUAD'  # ./*.jpeg
	LUSC_Patches_FROM =  '/scratch/nsr3/TCGA-LUSC/TCGA-LUSC'  # ./*.jpeg
	Normal_Patches_FROM = '/scratch/nsr3/normal-lung-cancer'  # ./*.jpeg

	TO_PATH = '/scratch/xc965/DL/TransLearn/data/data_pathology_small'

	if not os.path.exists(TO_PATH):
		os.makedirs(TO_PATH)


	for i in ['train', 'val', 'test']:
		create_folder = os.path.join(TO_PATH, i)
		if not os.path.exists(create_folder):
			os.makedirs(create_folder)


	for folder in [LUAD_Patches_FROM, LUSC_Patches_FROM, Normal_Patches_FROM]:

		if folder.split('-')[-1] == 'LUAD':
			cat = 'LUAD'  # TCGA-LUAD
		elif folder.split('-')[-1] == 'LUSC':
			cat = 'LUSC'  # TCGA-LUSC
		else:
			cat = 'NORM'  # normal-lung-cancer

		# Create the structured folders
		create_folder_list = []

		for i in ['train', 'val', 'test']:
			create_folder_list.append(os.path.join(TO_PATH, i))

		for path in create_folder_list:
			new_folder = os.path.join(path, cat)
			if not os.path.exists(new_folder):
				os.makedirs(new_folder)

		# Split images into train-val-test sets and save
		train_images, val_images, test_images = [], [], []
		for image in os.listdir(folder):
			if image.endswith('jpeg'):
				# parse the images into train-val-test folders
				if image.startswith('train'):
					if len(train_images) >= 70:
						continue
					train_images.append(image)
				elif image.startswith('valid'):
					if len(val_images) >= 10:
						continue
					val_images.append(image)
				elif image.startswith('test'):
					if len(test_images) >= 20:
						continue
					test_images.append(image)
				else:
					pass

		for image_list in [train_images, val_images, test_images]:
			random.shuffle(image_list)
			print(len(image_list))

		# split_index = floor(len(image_list) * split)
		# train_images = image_list[: split_index]
		# val_images= image_list[split_index :]

		for image in train_images:
			src = os.path.join(folder, image)
			dst = os.path.join(TO_PATH+'/train', cat)+'/'+image
			shutil.copy(src, dst)

		for image in val_images:
			src = os.path.join(folder, image)
			dst = os.path.join(TO_PATH+'/val', cat)+'/'+image
			shutil.copy(src, dst)

		for image in test_images:
			src = os.path.join(folder, image)
			dst = os.path.join(TO_PATH+'/test', cat)+'/'+image
			shutil.copy(src, dst)


if __name__ == '__main__':
	main()
