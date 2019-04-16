# VAE -- Variational Autoencoder

Implementation of VAE model, following the paper: [VAE](https://arxiv.org/abs/1312.6114). The encoder and decoder functions are implemented using fully strided convoluttional layers and transposed convolution layers respectively. As suggested by authors we have implemented Gaussian decoders and Gaussian prior in this work.

## Setup
* Python 3.5+
* Tensorflow 1.9

## Relevant Code Files

File config.py contains the hyper-parameters for VAE reported results.

File vae.py contains the code to train VAE model.

Similarly, as the name suggests, file vae_inference.py contains the code to test the trained VAE model.

## Usage
### Training a model
NOTE: For celebA, make sure you have the downloaded dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/Celeb) and keep it in the current directory of project.
```
python vae.py
```

### Test a trained model 
 
First place the model weights in model_directory (mentioned in vae_inference.py) and then:
```
python vae_inference.py 
```

## Empirical Observations

* We observe same inferences for KL weight as discussed [https://github.com/PrateekMunjal/VAE_GAN/blob/master/README.md](here).

* Although VAE has a more stable behaviour than VAE/GAN. However, the quality of VAE generations are less plausible than VAE/GAN generations.

* Essentially, the generations of VAE are blurry i.e it does not capture the small details of generated people.

* Blurry issue of VAE is explained by following:
  * The l2 loss function used for reconstruction loss.
  * There is no explicit term penalizing the decoder for generating non-plausible images. -- Alleviating this by an adversary gives birth to VAE/GAN model. Check my implementation for [https://github.com/PrateekMunjal/VAE_GAN](VAE/GAN).
```  
Why minimizing the l2 loss function produces blurred images?
```
After a bit of math, minimizing the l2 loss boils down to minimize the forward-KL divergence. The issue with forward-KL divergence is that it does not penalizes the model when our model puts weight on the region of data which is disjoint from the support of true distribution. 

## Generations

MNIST            |  Celeb-A
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/VAE/blob/master/MNIST/generations.gif)  |  ![](https://github.com/PrateekMunjal/VAE/blob/master/celebA/generations.gif)

## Reconstructions

MNIST Original            |  MNIST Reconstruction
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/VAE/blob/master/MNIST/op-real/original_new_vae-95.png)  |  ![](https://github.com/PrateekMunjal/VAE/blob/master/MNIST/op-recons/reconstructed_new_vae-95.png)

Celeb-A Original            |  Celeb-A Reconstruction
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/VAE/blob/master/celebA/op-real/orig-img-8.png)  |  ![](https://github.com/PrateekMunjal/VAE/blob/master/celebA/op-recons/recons-img-8.png)

