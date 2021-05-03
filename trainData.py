import numpy as np
from torch.utils import data
import torch.optim as optim


from VAE import *


def reconstruction_function(y, x):
    nn.MSELoss(size_average=False)
    pass


class Regression:
    gpu = 0
    batch_size = 1
    num_workers = 4
    niters = 200
    val_freq = 5
    learning_rate = 0.001

    best_val_loss = None
    best_val_itr = None

    def __init__(self, linear_space, latent_space):
        self.k_size = linear_space
        self.r_size = latent_space
        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.create(linear_space, latent_space)
        # self.loss_fn = None
        return

    def init_train(self, datasets=None, fold=None, niters=10, valfreq=5, lr=0.001):
        self.datasets = datasets
        self.fold = fold
        self.niters = niters
        self.val_freq = valfreq
        self.learning_rate = lr

        # data generator
        params_train = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.num_workers}
        params_test = {'batch_size': self.batch_size, 'shuffle': False, 'num_workers': self.num_workers}
        self.generators = {'train_data': data.DataLoader(self.datasets['train_data'], **params_train),
                           'val_data': data.DataLoader(self.datasets['val_data'], **params_train),
                           'test_data': data.DataLoader(self.datasets['test_data'], **params_test)}

        self.reconstruction_function = torch.nn.MSELoss(size_average=False)

    def loss_function(self, y, x, mu, log_var):

        BCE = self.reconstruction_function(y, x)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        #print('BCE={:.4f} KLD={:.4f} BCE+KLD={:.4f}'.format(BCE, KLD, BCE + KLD))
        return BCE + KLD

    def train(self, mu=None, log_var=None):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # ToDo initialize loss list

        training_loss = []

        for epoch in range(1, self.niters + 1):
            print('Training: Epoch={}'.format(epoch))
            # optimize zero gradients
            self.model.train()
            optimizer.zero_grad()

            train_epoch_loss = []
            # VAE for training
            for _, x in self.generators['train_data']:
                #print(x.shape, 'x input')
                # send x to GPU
                x = x.to(self.device)
                # split into 9 torch arrays (for the vae creating 9 predictions)(squeeze)
                x = torch.squeeze(x)
               #print(x.shape)

                # predictions
                predictions, mu, log_var = self.model(x.float())
                #print(predictions.shape, 'predictions')
                predictions = predictions.reshape(9, self.k_size)

                # compute loss
                comp_loss = self.loss_function(predictions, x.float(), mu, log_var)
                train_epoch_loss.append(comp_loss.item())

                # back propagation loss
                comp_loss.backward()

                # optimize.step
                optimizer.step()

            train_epoch_loss = np.mean(train_epoch_loss)

            # ToDo create list of loss append
            training_loss.append(train_epoch_loss)

            # find best loss (best weights)
            if epoch == 1:
                self.best_val_loss = train_epoch_loss
                self.best_val_itr = epoch
                self.model.save_state()

            if epoch % self.val_freq == 0:
                print('Validating: Epoch={}'.format(epoch))
                # VAE for eval
                self.model.eval()
                # with no grad
                with torch.no_grad():
                    val_epoch_loss = []
                    for _, x in self.generators['val_data']:
                        # Send x to gpu
                        x = x.to(self.device)
                        x = torch.squeeze(x)
                        # predictions(may include mu/log_var)
                        predictions, mu, log_var = self.model(x.float())
                        predictions = predictions.reshape(9, self.k_size)

                        # compute loss
                        comp_loss = self.loss_function(predictions, x.float(), mu, log_var)
                        val_epoch_loss.append(comp_loss.item())

                    val_epoch_loss = np.mean(val_epoch_loss)

                    # testing best loss
                    if val_epoch_loss < self.best_val_loss:
                        self.best_val_loss = val_epoch_loss
                        self.best_val_itr = epoch
                        self.model.save_state()
                        print('     [INFO] checkpoint with val_epoch_loss={} saved!'.format(val_epoch_loss))
        # ToDo return train_loss
        return training_loss

    def test(self):
        print('Testing')
        self.model.load_state()
        self.model.eval()
        n_counter = 0
        #print('................... cases for testing = {}'.format(self.datasets['test_data'].__len__()))
        all_predictions = np.zeros((self.datasets['test_data'].__len__(), 9, self.k_size), dtype=np.float32)
        with torch.no_grad():
            test_loss = []
            for index, x in self.generators['test_data']:
                n_counter+=1
                # send s to gpu
                x = x.to(self.device)
                x = torch.squeeze(x)

                #print('index={}, test data[{}]'.format(index, x.shape))

                # test predict
                predictions, mu, log_var = self.model(x.float())
                predictions = predictions.reshape(9, self.k_size)
                all_predictions[index] = predictions.detach().cpu().numpy()

                # compute loss
                comp_loss = self.loss_function(predictions, x.float(), mu, log_var)
                test_loss.append(comp_loss.item())

            test_loss = np.mean(test_loss)

            #print(test_loss, 'Test loss')
            #print('n_counter', n_counter)
        return all_predictions, test_loss

    def create(self, linear_space, latent_space):
        model = VAE(linear_space, latent_space)
        if torch.cuda.is_available():
            model.cuda()
        return model
