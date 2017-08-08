ahow_many_datasets = 10
prefix = 'dataset_'
suffix = '.json'
original_fiename = 'annotations.json'
size = []
for i in range(how_many_datasets):
	size.append(0)
	open(prefix+str(i+1)+suffix, 'w')
with open(original_fiename, 'r') as original_f:
	line_no  = 0
	for line in original_f:
		line_no = line_no+1
		for i in range(how_many_datasets):
			if( line_no %(i+1) == 0 ):
				with open(prefix+str(i+1)+suffix, 'a') as d_f:
					d_f.write(line)
					size[i] = size[i]+1
with open('file_size.txt', 'w') as size_f:
	size_f.write(str(size).replace('[','').replace(']', ''))
