import torch
import torch.nn as nn
import torch.nn.functional as f

class VAE(nn.Module):
    best_parameters = None

    def __init__(self, linear_space, latent_space):
        super(VAE, self).__init__()

        average = (linear_space + latent_space)/2
        average = round(average)

        #encoder
        nn.Flatten()
        self.enc1 = nn.Linear(in_features = linear_space, out_features = average)
        self.enc2 = nn.Linear(in_features = average, out_features = latent_space)
        self.enc3 = nn.Linear(in_features=average, out_features=latent_space)

        #decoder
        self.dec1 = nn.Linear(in_features = latent_space, out_features = average)
        self.dec2 = nn.Linear(in_features = average, out_features = linear_space)

    #encoder class
    def encode(self, x):
        h1 = f.relu(self.enc1(x))
        return self.enc2(h1), self.enc3(h1)

    # Reparameterization - encoder output mean and variance
    def reparameterize(self, mu, log_var):
        #standard deviation
        std = torch.exp(0.5*log_var)
        #provides the same size as standard deviation
        eps = torch.randn_like(std)
        # sampling the latent space
        z = mu + (eps*std)
        return z

    #decoder class
    def decode(self, z):
        h3 = f.relu(self.dec1(z))
        return f.sigmoid(self.dec2(h3))

    #x is the input data
    def forward(self, x):
        #encoding - using the leaky_relu makes the encoder non-linear and able
        #to map more complex patterns. leaky_relu vs relu includes some negative values too
        #allowing for greater variance??

        x=x.view(x.size(0), -1)
        #print(x.shape,'x for')
        mu, logvar = self.encode(x)

        #latent vector from reparameterisation
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def load_state(self):
        self.load_state_dict(self.best_parameters)

    def save_state(self):
        self.best_parameters = self.state_dict()
