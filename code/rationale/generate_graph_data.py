import os

#lamda_1 = [0.0002]
#lamda_1 = [0.0003]
#lamda_1 = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005, 0.00001, 0.00005, 0.00009, 00012, 0.00016, 0.000005]

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]# 0.000085, 0.000095, 0.0001, 0.00016, 0.00025, 0.0004
#lamda_1 = [ 0.000115] # for small 0.000085, 2 gru, markov, 
lamda_2 = [ 1, 2]
dropout = [0.1]
max_epochs = 50
aspect = 1
select_all = -1
output_file = 'outputs_2.json'
graph_data_file = 'graph_data/data_vs_lamda_table_3Par.txt'
open(graph_data_file, 'w')
for l_1 in lamda_1:
	for l_2 in lamda_2:
		for dp in dropout:
			for _,cur_epoch in enumerate(range(0, 65, 5)):
				model_file = 'models/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)#+'_epochs_'+str(trained_max_epochs) #+'.txt.pkl.gz'
				#load_model_file = 'models/'+model_file  +'_epochs_'+str(trained_max_epochs)+'.txt.pkl.gz'
				model_file = model_file+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'+str(cur_epoch)+'.pkl.gz'
				#model_file = 'model_sparsity_'+str(l_1)+'_epochs_'+str(max_epochs)+'.txt.pkl.gz'
				#model_file = 'models/'+model_file
				if not os.path.exists(model_file): 
					continue
					#print 'not exist: ', model_file
				run_command = 'THEANO_FLAGS="mode=FAST_RUN,device=gpu2,floatX=float32" python dev.py --embedding word_vec.gz --load_rationale annotations.json --dump outputs_with_first_loading.json --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + model_file  + ' --graph_data_path '+ graph_data_file + ' --cur_epoch '+ str(cur_epoch)
				#print run_command
				os.system(run_command)
				print '\n\n\n '
				#exit()
