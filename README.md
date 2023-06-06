# Gaussian Mixture Variational AutoEncoders (GMVAE) for MD analysis #

This code is an updated version of the GMVAE implementation by Yasemin Bozkurt Varolgunes, which can be found https://github.com/yabozkurt/gmvae. 
Changes have been made to ensure compatibility with TensorFlow 2.

The following text was obtained from the article: Interpretable embeddings from molecular simulations using Gaussian mixture variational autoencoders 
(https://doi.org/10.1088/2632-2153/ab80b7)

GMVAE architecture and training hyperparameters:

The GMVAE algorithm is implemented in Tensorflow 2.5.0. Training was performed in all cases with fully-connected layers, 
using the Adam optimization algorithm. The Softmax activation function was used for probabilistic cluster assignments, 
while ReLu activation functions were employed in all hidden layers. The means were obtained without any activation, 
whereas Softplus activation was employed to obtain the variances. Below shows the hyperparameters and their default values used. 
Default values were employed wherever the parameters are not specified. The NN(·)’s correspond to the neural networks labeled. 
NN(Qy) performs probabilistic cluster assignments, NN(Qz) is for learning the moments of each Gaussian distribution in the encoding, 
whereas NN(Pz) and NN(Px) are for the decoding of the z and x, respectively.
The lengths of the ‘Number of nodes’ entries correspond to the number of hidden layers. Hyperparameter optimization was carried out as follows. 
The number of nodes in the decoder (NN(Px)) was then increased whenever a large and non-decreasing reconstruction loss was observed. 
In other studies the overall observation for the previous  examples is that the learning rate 
and batch size should be kept relatively low to promote the formation of distinct clusters. The VAE results (with unimodal Gaussian prior) 
that are provided as comparison are obtained using k = 1, while keeping the remaining parameters equal to the values in the corresponding GMVAE model.

Understanding the Algorithm:

First, data points are probabilistically assigned to k clusters (NN(Qy)). Q(y|x) represents these cluster assignment probabilities, 
and has a multinomial distribution. Since each cluster is assumed to have Gaussian distribution in the latent space, the mean and
variance of each of these Gaussians (Q(z|x, y)) are learned via the encoder part of the neural network (NN(Qz)). 
The low-dimensional representation, z, is then obtained by first sampling and then taking the expected value of these samples, 
i.e. z = sum_{i=0}^{k-1} (p(y_i|x)z_i. As the first step in decoding, the moments of the corresponding low-dimensional representation z 
is learned by NN(Pz) from each Gaussian-distributed individual cluster y_i , which is then followed by a sampling operation. 
P(y) in the decoder is assumed to be uniformly distributed among the k clusters. Next, using the encodings, z_i ’s, 
the associated x reconstructions are obtained again by sampling from the x' by the NN(Px). Similar to the encoder, 
the decoder obtains a fixed reconstruction by taking the expected value of x'_i ’s.

HYPERPARAMETERS:										
												
  '--traj_file'  :  Path to trajectory (npy file)						
  '--k'          :  Number of mixture components (default=5)							
  '--n_x'        :  Number of observable dimensions (reconstruction dimensions = 3)			
  '--n_z'        :  Number of hidden dimensions (default = 1)					
  '--n_epochs'   :  Number of epochs (default = 50)						
  '--qy_dims'    :  Iterable of hidden dimensions in qy subgraph (default = 32)			
  '--qz_dims'    :  Iterable of hidden dimensions in qz subgraph (default = 16)			
  '--pz_dims'    :  Iterable of hidden dimensions in pz subgraph (default = 16)				
  '--px_dims'    :  Iterable of hidden dimensions in px subgraph (default = 128			
  '--r_nent'     :  Constant for weighting negative entropy term in the loss (default = 0.05)	
  '--batch_size' :  Number of samples in each batch (default = 517)				
  '--lr'         :  Learning rate (default = 0.0001)					


Dependencies and Modules (CTE-Power - BSC-CNS HPC):
    
   module load atlas/3.10.3 
	       scalapack/2.0.2 
               szip/2.1.1  
               gcc/8.3.0 
               cuda/10.2 
               cudnn/7.6.4 
               nccl/2.4.8 
               tensorrt/6.0.1 
               openmpi/4.0.1 
               fftw/3.3.8 
               ffmpeg/4.2.1 
               opencv/4.1.1 
               python/3.7.4_ML 
               tensorflow/2.5.0 
               anaconda3/2020.02
