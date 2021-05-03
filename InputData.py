import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from DataSets import DBSdataSet
from trainData import Regression
from scipy.linalg import svd

warnings.filterwarnings("ignore")

def main():
    datasets = {}
    cvResults = {'train_idx': [], 'val_idx': []}

    read_files = glob.glob("/Users/JoeKelly/PycharmProjects/SphereVAE/DBS100/*.vol")

    with open("allData.txt", "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

    DeleteLines = 'VM'

    with open('allData.txt') as oldfile, open('FinalData.txt', 'w') as FinalData:
        for line in oldfile:
            if not any(DeleteLine in line for DeleteLine in DeleteLines):
                FinalData.write(line)

    FinalData = pd.read_csv('FinalData.txt', delimiter=" ", names=["index", "x", "y", "z"])
    FinalData = FinalData.loc[0:, "x":"z"]

    df = pd.DataFrame(FinalData)
    arry = df.to_numpy()
    x = np.reshape(arry, (100, 10, 236, 3))
    x = x[:, 1:, :, :] - x[:, 0:-1, :, :]
    x = np.reshape(x, (100, 9, 708))

    # test/train
    N = len(x)
    idx = np.arange(N, dtype=np.int64)
    test_indices = np.random.choice(idx, 10, replace=False)
    test_indices = np.sort(test_indices)

    train_indices = np.setdiff1d(idx, test_indices)

    kf = KFold(n_splits=5, shuffle=True)

    lin_space = [64]
    latent_space = [36]


    n = 708
    T = 900
    mse_ps_mean, mse_ps_std, mseVAE_ps_mean, mseVAE_ps_std = [], [], [], []
    log_train, log_trainVAE = [], []
    for k in lin_space:
        for r in latent_space:
            cv_fold = 1
            mse_folds, mseVAE_folds = [], []
            train_loss_list, trainVAE_loss_list  = [], []
            for train_idx, val_idx in kf.split(train_indices):

                print('>>>>>>>>>>linear={} latent={} fold={}<<<<<<<<<<<'.format(k, r, cv_fold))

                x_train = x[train_indices[train_idx]]
                x_train = np.reshape(x_train, (len(train_idx)*9, 708))
                x_val = x[train_indices[val_idx]]
                x_val = np.reshape(x_val, (len(val_idx) * 9, 708))
                x_test_1 = x[test_indices]
                x_test = np.reshape(x_test_1, (len(test_indices) * 9, 708))

                # TODO visualise inputs
                # plt.hist(np.reshape(x[0, :, :], (9 * 708)), bins=50)

                _, _, VT_train = svd(x_train, full_matrices=False)

                U_train = VT_train[:k].T
                q_train1 = x[train_indices[train_idx]] @ U_train
                q_train = np.reshape(q_train1, (len(train_idx), 9, k))


                datasets['train_data'] = DBSdataSet(data=q_train, indices=None)

                _, _, VT_val = svd(x_val, full_matrices=False)
                U_val = VT_val[:k].T
                q_val = x[train_indices[val_idx]] @ U_val
                q_val = np.reshape(q_val, (len(val_idx), 9, k))


                datasets['val_data'] = DBSdataSet(data=q_val, indices=None)

                _, _, VT_test = svd(x_test, full_matrices=False)
                U_test = VT_test[:k].T
                q_test = x[test_indices] @ U_test
                q_test = np.reshape(q_test, (len(test_indices), 9, k))


                datasets['test_data'] = DBSdataSet(data=q_test, indices=None)


                learning = Regression(k, r)
                learning.init_train(datasets=datasets, fold=cv_fold, niters=20, valfreq=5, lr=0.001)

                train_loss = learning.train()

                q_test_hat, test_lost = learning.test()

                u_hat_1 = q_test_hat@U_test.T


                u_hat = np.reshape(u_hat_1, (len(test_indices)*9, 708))


                mse = mean_squared_error(x_test, u_hat)

                mse_folds.append(mse)
                train_loss_list.append(train_loss)
                mean_train_loss = np.mean(train_loss_list, axis=0)


                print('>>>>>>>>>>>>VAE only training Latent={} fold={}<<<<<<<<<<'.format(r, cv_fold))
                datasets['train_data'] = DBSdataSet(data=x, indices=train_indices[train_idx])
                datasets['val_data'] = DBSdataSet(data=x, indices=train_indices[val_idx])
                datasets['test_data'] = DBSdataSet(data=x, indices=test_indices)

                learningVAE = Regression(n, r)
                learningVAE.init_train(datasets=datasets, fold=cv_fold, niters=20, valfreq=5, lr=0.001)
                train_loss_VAE = learningVAE.train()

                x_test_VAE, test_lost_VAE = learningVAE.test()

                # TODO have ready inputs and predictions
                # x_test = np.random.normal(0, 0.1, (10*9, 708))
                # xh_test_vaeonly = np.random.normal(0, 0.1, (10*9, 708))
                # xh_test_vaesvd = np.random.normal(0, 0.1, (10*9, 708))

                # TODO non-zero indices
                # threshold = 0.001
                # x_test_nz_cond = np.abs(x_test) > threshold
                # xh_test_vaeonly_nz_cond = np.abs(xh_test_vaeonly) > threshold
                # xh_test_vaesvd_nz_cond = np.abs(xh_test_vaesvd) > threshold

                # TODO non-zero values
                # x_test_nz = x_test[x_test_nz_cond]
                # x_test_nz_vaeonly = xh_test_vaeonly[x_test_nz_cond]
                # x_test_nz_vaesvd = xh_test_vaesvd[x_test_nz_cond]

                # TODO MSE of non-zero values only
                # mean_squared_error(x_test_nz, x_test_nz_vaeonly)
                # mean_squared_error(x_test_nz, x_test_nz_vaesvd)
                # x_test_nz.shape how many?

                # TODO save/load
                # import pickle
                # pickle.dump(x_test, open('./input.pkl','wb'))
                # x_loaded = pickle.load(open('./input.pkl','rb'))


                mseVAE = mean_squared_error(x_test, x_test_VAE.reshape(len(test_indices)*9, 708))

                mseVAE_folds.append(mseVAE)
                trainVAE_loss_list.append(train_loss_VAE)
                mean_trainVAE_loss = np.mean(trainVAE_loss_list, axis=0)
                #print(trainVAE_loss_list, 'VAE loss list')
                print(mean_trainVAE_loss, 'mean train VAE loss list')



                print(mse, 'MSE' )
                print(mseVAE, 'VAE MSE')



                cv_fold += 1
                # input('wait')


            # plt.figure(2)
            # plt.plot(np.linspace(1, len(train_loss), len(train_loss)), mean_train_loss)
            # plt.title('VAE only: linear={} latent={}: Average over all folds'.format(k, r, cv_fold))
            # plt.xlabel('Epochs')
            # plt.ylabel('Training Loss')
            # plt.draw()
            # plt.show()



            mse_fold_mean, mse_fold_std = np.mean(mse_folds), np.std(mse_folds)
            mse_ps_mean.append(mse_fold_mean)
            mse_ps_std.append(mse_fold_std)

            log_train_min = np.log(min(mean_train_loss))
            log_train.append(log_train_min)
            #
            mseVAE_fold_mean, mseVAE_fold_std = np.mean(mseVAE_folds), np.std(mseVAE_folds)
            mseVAE_ps_mean.append(mseVAE_fold_mean)
            mseVAE_ps_std.append(mseVAE_fold_std)

            threshold = 0.0001
            x_t1 = x_test_1[0]
            x_t1 = np.reshape(x_t1, (1, 6372))
            x_t1 = np.squeeze(x_t1)

            x_test_nz_cond = np.abs(x_t1) > threshold
            print(x_test_nz_cond.shape, 'total')

            x_disp = x_t1[x_test_nz_cond]
            print(x_disp.shape, 'xdisp shape')

            x_hat_nonzeros = u_hat_1[0]
            x_hat_nonzeros = np.reshape(x_hat_nonzeros, (1, 6372))
            x_hat_nonzeros = np.squeeze(x_hat_nonzeros)

            x_hat_nonzeros = np.array(x_hat_nonzeros)
            x_hat_nonzeros = x_hat_nonzeros[x_test_nz_cond]

            print(x_hat_nonzeros.shape, 'x_hat_nonzeros')


            x_hat_nonzeros_VAE = x_test_VAE[0]
            x_hat_nonzeros_VAE = np.reshape(x_hat_nonzeros_VAE, (1, 6372))
            x_hat_nonzeros_VAE = np.squeeze(x_hat_nonzeros_VAE)

            x_hat_nonzeros_VAE = np.array(x_hat_nonzeros_VAE)
            x_hat_nonzeros_VAE = x_hat_nonzeros_VAE[x_test_nz_cond]
            print(x_hat_nonzeros_VAE.shape, 'x_hat_nonzeros_VAE')


        # print('MSE (test loss) across folds for each parameter space')
        print('means = {}'.format(mse_ps_mean))
        # print('standard deviations = {}'.format(mse_ps_std))
        # print(log_train, 'log train values')

        # print('VAE MSE (test loss) across folds for each parameter space')
        print('VAE means = {}'.format(mseVAE_ps_mean))
        # print('VAE standard deviations = {}'.format(mseVAE_ps_std))

    # linsp = [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51]
    # latsp = [[60, 50, 40, 30, 20, 10]]
    #
    # loss = log_train
    # df=pd.DataFrame({'latsp':latsp, 'linsp': linsp, 'loss':loss})
    # df_wide = df.pivot_table(index='latsp', columns='linsp', values='loss')
    # sns.heatmap(df_wide)
    # plt.show()


    mse_adjusted= mean_squared_error(x_disp, x_hat_nonzeros)
    mse_adjusted_VAE = mean_squared_error(x_disp, x_hat_nonzeros_VAE)

    print(mse_adjusted, 'MSE adjusted')
    print(mse_adjusted_VAE, 'MSE VAE adjusted')


    plt.figure(1)
    plt.hist(x_t1, range= [-0.001, 0.001], bins=50)
    plt.title('Original Displacements')
    plt.xlabel('Displacements (mm)')
    plt.ylabel('Frequency')
    plt.figure(2)
    plt.hist(x_hat_nonzeros, range= [-0.001, 0.001], bins=50)
    plt.title('SVD+VAE')
    plt.xlabel('Displacements (mm)')
    plt.ylabel('Frequency')
    plt.figure(3)
    plt.hist(x_hat_nonzeros_VAE, range= [-0.001, 0.001], bins=50)
    plt.title('VAE-only')
    plt.xlabel('Displacements (mm)')
    plt.ylabel('Frequency')

    plt.show()

    data = [x_t1, x_disp, x_hat_nonzeros, x_hat_nonzeros_VAE]

    pickle.dump(data, open('./input.pkl','wb'))
    #x_loaded = pickle.load(open('./input.pkl','rb'))


if __name__ == '__main__':
    main()
