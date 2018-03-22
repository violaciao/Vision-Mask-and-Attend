import os
import random
import shutil
from math import floor


def main():
	random.seed(1)
	split = 0.8

	FROM_PATH = '/Users/Viola/Documents/workspace/flower/downloaded/flower_5'
	TO_PATH = '/Users/Viola/CDS/Rearch/Langone/Vision-Mask-and-Attend/data_flower5'


	for i in ['train', 'val']:
		create_folder = os.path.join(TO_PATH, i)
		if not os.path.exists(create_folder):
			os.makedirs(create_folder)


	for folder in os.listdir(FROM_PATH):

		if folder.endswith('txt') == 0:

			# Create the structured folders
			create_folder_list = []

			for i in ['train', 'val']:
				create_folder_list.append(os.path.join(TO_PATH, i))

			for path in create_folder_list:
				new_folder = os.path.join(path, folder)
				if not os.path.exists(new_folder):
					os.makedirs(new_folder)

			# Split images into train-val sets and save
			image_list = []
			for image in os.listdir(os.path.join(FROM_PATH, folder)):
				image_list.append(image)

			random.shuffle(image_list)
			print(len(image_list))

			split_index = floor(len(image_list) * split)
			train_images = image_list[: split_index]
			val_images= image_list[split_index :]

			for image in train_images:
				src = os.path.join(os.path.join(FROM_PATH, folder), image)
				dst = os.path.join(TO_PATH+'/train', folder)+'/'+image
				shutil.copy(src, dst)

			for image in val_images:
				src = os.path.join(os.path.join(FROM_PATH, folder), image)
				dst = os.path.join(TO_PATH+'/val', folder)+'/'+image
				shutil.copy(src, dst)



if __name__ == '__main__':
	main()
