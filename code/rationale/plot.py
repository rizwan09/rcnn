import argparse
import os
import numpy as np


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
# plt.style.use('bmh')


def plot_enc_data_graph(x, y, y_label, g_label, name, args):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(x, y, '-gD', label = g_label)
    plt.xlabel('% of selected data size')
    #plt.xticks(np.arange(min(x), max(x) + 1))
    plt.ylabel(y_label)
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, name+'.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

def plot_mse_data_graph(x, y, y_label, g_label, name, args):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(x, y, 'D', label = g_label)

    #plt.plot(x+4, y, '>', label = g_label)
    plt.xlabel('% of selected data size')
    #plt.xticks(np.arange(min(x), max(x) + 1))
    plt.ylabel(y_label)
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, name+'.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))



def plot_mse_data_graph_scatter(x_list, y_list, y_label, g_label, name, args):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    for i in x_list:
        print x_list[i]
        print y_list[i]
        plt.plot(x_list[i], y_list[i], 'D', label = g_label)
        
   # plt.plot(x_list[0], y_list[0], 'D', label = g_label)

    #plt.plot(x_list[1], y_list[1], '>', label = g_label)
    plt.xlabel('% of selected data size')
    #plt.xticks(np.arange(min(x), max(x) + 1))
    plt.ylabel(y_label)
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, name+'.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))    

def plot_gen_data_graph(x, y, y_label, g_label, name, args):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(x, y, '-gD', label = g_label)
    plt.xlabel('% of selected data size')
    #plt.xticks(np.arange(min(x), max(x) + 1))
    plt.ylabel(y_label)
    plt.legend()
    plt.ylim(0, 35)
    loss_fname = os.path.join(args.graph_data_folder, name+'.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_path",
            type = str,
            #default = 'graph_data/data_vs_lamda_table _Copy.txt'
            #default = 'graph_data/data_vs_lamda_table.txt'
            default = '../graph_data/data_vs_lamda_table_3Par.txt'
        )
    parser.add_argument("--graph_data_folder",
            type = str,
            default = '../graph_data'
        )
    args = parser.parse_args()

    
    trainData = np.loadtxt(args.graph_data_path, delimiter='\t')
    r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, tmp_t, total_test_time, l_1, l_2, m_e, e = np.split(trainData, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1)
    
    size = []
    with open('file_size.txt', 'r') as size_f:
        for line in size_f:
            for word in line.split(','):
                size.append(int(word))
    
   
    r_p1 =  np.concatenate(np.dot(r_p1, 100), axis=0) 
    r_mse = np.concatenate(r_mse, axis=0) 
    gen_time =  np.concatenate(gen_time, axis=0)
    enc_time = np.concatenate(enc_time, axis = 0)
    total_test_time = np.concatenate(total_test_time, axis = 0)
    l_1 = np.concatenate(l_1, axis = 0)
    l_2 = np.concatenate(l_2, axis = 0)
    e =  np.concatenate(e, axis = 0)
    #print r_mse, r_p1

    size = r_p1
    assert len(size) == len(enc_time)
    print size, r_mse

    
    percentage = { s: [] for s in set(l_1)}
    mse = {s: [ ] for s in set(l_1)}

    for i in range(len(l_1)):
        percentage[l_1[i]] = r_p1[i]
        mse[l_1[i]] = r_mse[i]



    plot_mse_data_graph_scatter(percentage, mse, y_label = 'LOSS', g_label = 'mse vs % of selected data size', name = 'colored_mse_vs_%data_selection', args = args)
   
    #plot_enc_data_graph(size, enc_time, y_label = 'Seconds', g_label = 'enc time vs % of selected data size', name = 'enc_time_vs_%data_selection', args = args)
    
    #plot_gen_data_graph(size, gen_time, y_label = 'Seconds', g_label = 'gen time vs % of selected data size', name = 'gen_time_vs_%data_selection', args = args)
    
    plot_mse_data_graph(size, r_mse, y_label = 'LOSS', g_label = 'mse vs % of selected data size', name = 'mse_vs_%data_selection', args = args)
   

   #plot_graph(r_p1, gen_time, y_label = 'Generation time', g_label = 'Generation time vs selection', name = 'Generation_time', args = args)
    #plot_graph(r_p1, enc_time, y_label = 'Encoder time', g_label = 'Encoder time vs selection', name = 'Encoding_time', args = args)
    #plt.plot(r_p1, gen_time, label = 'Genration time vs percentage selection')
    #plt.plot(r_p1, enc_time, label = 'Encoder time vs percentage selection')
    #plt.plot(r_p1, total_test_time, label = 'Total time vs percentage selection')

    


    
    

if __name__ == '__main__':
    main()
