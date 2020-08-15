from kaldiio import ReadHelper
import numpy as np 
import tqdm, glob, os
import argparse


class load_data(object):
	def __init__(self, args):
		self.save_path = args.xvector_npy_path
		self.vector_files = glob.glob('%s/xvector*.scp'% args.xvector_path)
		self.main_folder=args.main_folder+'/'
		for file in tqdm.tqdm(self.vector_files):
			self.get_npy(file)

	def get_npy(self, scp_file):
		with ReadHelper('scp:' + scp_file) as reader:
			for key, array in reader:
				npy_save_path = self.save_path + self.main_folder + key.replace('-','/') + '.npy'
				if not os.path.exists(self.save_path + self.main_folder + key.split('-')[0]):
					os.makedirs(self.save_path + self.main_folder + key.split('-')[0])
				np.save(npy_save_path, array)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='LRS2 dataset')
	parser.add_argument('--xvector_path', default= 'exp/Audio_2mix_min/pretrain/Xvector',type=str)
	parser.add_argument('--xvector_npy_path', default="/home/panzexu/datasets/tmp/", type=str)
	parser.add_argument('--main_folder', default='main',type=str)
	args = parser.parse_args()

	load_data(args)