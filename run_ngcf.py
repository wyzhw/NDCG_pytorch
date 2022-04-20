'''
Run this file in terminal with arguments, per example:
>> run.py --batch_size 512 --mess_dropout 0. --k 10 --n_epochs 5 --threshold 2
'''

import pandas as pd
import torch

import os
from time import time
from datetime import datetime
from matplotlib import pyplot as plt

from utils.load_data import Data
from utils.parser import parse_args
from utils.helper_functions import early_stopping,\
                                   train,\
                                   split_matrix,\
                                   compute_ndcg_k,\
                                   eval_model
from utils.datapreprocessing import create_data
from ngcf import NGCF

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

if __name__ == '__main__':

    # read parsed arguments
    args = parse_args()
    data_dir = args.data_dir # default='./data/'
    dataset = args.dataset
    batch_size = args.batch_size # Store model to path default='results',
    layers = eval(args.layers) # default='[64,64]'
    emb_dim = args.emb_dim # 'number of embeddings. default=64
    lr = args.lr
    reg = args.reg # default=1e-5, help='l2 reg.'
    mess_dropout = args.mess_dropout
    node_dropout = args.node_dropout
    k = args.k # 'k order of metric evaluation (e.g. NDCG@k)'
    n = args.threshold
    split_percentage = args.sp


    create_data(n, split_percentage)

    # generate the NGCF-adjacency matrix
    data_generator = Data(path=data_dir + dataset, batch_size=batch_size)
    adj_mtx = data_generator.get_adj_mat() # get an adjacency-matrix

    # create model name and save
    modelname =  "NGCF" + \
        "_bs_" + str(batch_size) + \
        "_nemb_" + str(emb_dim) + \
        "_layers_" + str(layers) + \
        "_nodedr_" + str(node_dropout) + \
        "_messdr_" + str(mess_dropout) + \
        "_reg_" + str(reg) + \
        "_lr_"  + str(lr)

    # create NGCF model
    model = NGCF(data_generator.n_users, 
                 data_generator.n_items,
                 emb_dim,
                 layers,
                 reg,
                 node_dropout,
                 mess_dropout,
                 adj_mtx)
    if use_cuda:
        model = model.cuda()

    # current best metric
    cur_best_metric = 0

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set values for early stopping
    cur_best_loss, stopping_step, should_stop = 1e3, 0, False
    today = datetime.now()

    print("Start at " + str(today))
    print("Using " + str(device) + " for computations")
    print("Params on CUDA: " + str(next(model.parameters()).is_cuda))

    results = {"Epoch": [],
               "Loss": [],
               "Recall": [],
               "NDCG": [],
               "Training Time": []}

    for epoch in range(args.n_epochs):

        t1 = time()
        loss = train(model, data_generator, optimizer)
        training_time = time()-t1
        print("Epoch: {}, Training time: {:.2f}s, Loss: {:.4f}".
            format(epoch, training_time, loss))

        # print test evaluation metrics every N epochs (provided by args.eval_N)
        if epoch % args.eval_N  == (args.eval_N - 1):
            with torch.no_grad():
                t2 = time()
                recall, ndcg = eval_model(model.u_g_embeddings.detach(),
                                          model.i_g_embeddings.detach(),
                                          data_generator.R_train,
                                          data_generator.R_test,
                                          k)
            print(
                "Evaluate current model:\n",
                "Epoch: {}, Validation time: {:.2f}s".format(epoch, time()-t2),"\n",
                "Loss: {:.4f}:".format(loss), "\n",
                "Recall@{}: {:.4f}".format(k, recall), "\n",
                "NDCG@{}: {:.4f}".format(k, ndcg)
                )
            #  make it stop if recall do not increase within 5 epochs
            cur_best_metric, stopping_step, should_stop = \
            early_stopping(recall, cur_best_metric, stopping_step, flag_step=5)

            # save results in dict
            results['Epoch'].append(epoch)
            results['Loss'].append(loss)
            results['Recall'].append(recall.item())
            results['NDCG'].append(ndcg.item())
            results['Training Time'].append(training_time)
        else:
            # save results in dict
            results['Epoch'].append(epoch)
            results['Loss'].append(loss)
            results['Recall'].append(None)
            results['NDCG'].append(None)
            results['Training Time'].append(training_time)

        if should_stop == True: break

    # save
    if args.save_results:
        date = today.strftime("%d%m%Y_%H%M")

        # save model as .pt file
        if os.path.isdir("./models"):
            torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")
        else:
            os.mkdir("./models")
            torch.save(model.state_dict(), "./models/" + str(date) + "_" + modelname + "_" + dataset + ".pt")

        # save results as pandas dataframe
        results_df = pd.DataFrame(results)
        results_df.set_index('Epoch', inplace=True)
        if os.path.isdir("./results"):
            results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")
        else:
            os.mkdir("./results")
            results_df.to_csv("./results/" + str(date) + "_" + modelname + "_" + dataset + ".csv")

        # plot loss
        #//subplot four graphs

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(results['Loss'], 'b-', label='Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(results['Recall'], 'r-', label='Recall')
        plt.plot(results['NDCG'], 'g-', label='NDCG')
        plt.title('Evaluation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()

    plt.savefig("./results/" + str(date) + "_" + modelname + "_" + dataset + ".png")
    plt.show()
