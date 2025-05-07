# diffusion_test_time_compute
TL;DR: We adapted DeepMind’s inference-scaling framework-originally designed to do noise space search at inference for deterministic ODE-based diffusion models—to work with stochastic DDPMs, turning the noise space search problem into a denoising trajectory space search problem. To enable test-time conditioning, we developed a custom verifier that leverages small subsets of labeled training examples, allowing us to guide an unconditioned model at inference. Despite limited compute resources, we successfully validated our approach on MNIST, demonstrating that minimal labeled data can effectively steer generative outputs. Moving forward, we aim to extend this framework to larger, more complex datasets and explore its application in robotics diffusion policies. Also, because of the nature of DDPMs and stochastic denoising, we believe that the most optimal method of search will be to distribute compute differently across different phases of inference, and so we will look into this as well. 

The future_generalization_experiment directory has an example of a generalization test we would like to run in the future with more complex datasets. This experiment would basically tell us if an image better fits the distribution of images used in the verifier or the overall distribution of images used for training the diffusion model.

To the best of our knowledge, this is the first-ever implementation of and research on inference scaling via denoising trajectory space search applied to stochastic diffusion models.

-----------------------------------------------------------------------------------------------------------------------------------------------
Directory Configuration:
* main_experiments/ - Code and classifier models used for experiments in the paper.
* lc_gpu_accelerated_training/ - Code to train label-conditioned diffusion model for MNIST on a GPU. .env with lambda ssh key and api key required in main repo dir. 
* nlc_gpu_accelerated_training/ - Code to train non-label-conditioned diffusion model for MNIST on a GPU. .env with lambda ssh key and api key required in main repo dir. 
* inference_experiments/ - Production grade code with utils, refactored from main_experiments for folks who want more generalized code.
* model_tests/ - Basic model tests conducted in prepration for larger experiments. 
* future_generalization_experiment/ - Code for experiments we would like to perform on larger datasets and models.