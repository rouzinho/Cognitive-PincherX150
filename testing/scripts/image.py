from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

#img = np.asarray(Image.open('/home/PhD/Codes/Experiment-IMVAE/datas/production/habituation/0/outcome_latent_space.npy'))
im = np.load("/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/habituation/0/outcome_latent_space.npy")
plt.imshow(im, cmap='gray')
plt.show()