import kaldiio, glob, numpy, tqdm

run='MuSE_1'

for fdr in ['test','train']:
	train_file = "/home/panzexu/Download/eer/runs/%s/%s"%(run,fdr)

	train_files = glob.glob("%s/*/*/*.npy"%train_file)
	print(len(train_files), train_files[0])

	dict_vector = {}
	for file in tqdm.tqdm(train_files):
		key = file.split('/')[-3] + '-' + file.split('/')[-2] + '-' + file.split('/')[-1].replace('.npy', '')
		mat = numpy.load(file)
		dict_vector[key] = mat

	print(len(dict_vector))

	with kaldiio.WriteHelper('ark,scp:feat/%s_%s_feat.ark,feat/%s_%s_feat.scp'%(run,fdr,run,fdr)) as writer:	
		for key in dict_vector.keys():
			mat = dict_vector[key]
			writer(key, mat)

# file = "/home/ruijie/kaldi/egs/voxceleb2/v2/feat/trials"
# file_o = "/home/ruijie/kaldi/egs/voxceleb2/v2/feat/new_trials"

# f_o = open(file_o, 'w')
# i = 0
# with open(file) as f:
# 	while True:
# 		line = f.readline()
# 		if not line:
# 			break
# 		line = line.replace('/', '-')
# 		f_o.write(line)
# 		i += 1
# 		if i % 100 == 0:
# 			print(i)