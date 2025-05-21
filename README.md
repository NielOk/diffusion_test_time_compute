# diffusion_test_time_compute
TL;DR: We present a novel framework for inference scaling in stochastic diffusion models beyond time steps, extending prior work from deterministic (ODE-based) noise space search to the stochastic setting of DDPMs. While prior efforts by DeepMind have framed inference-time optimization as a search over initial noise vectors in deterministic reverse processes, we generalize this to stochastic processes by introducing trajectory space search. This reframing treats the entire sequence of reverse-time noise injections as a structured object of optimization, enabling targeted exploration of the denoising landscape. We additionally develop a test-time conditioning mechanism via verifier-guided selection over sampled trajectories using a small labeled subset, enabling effective output steering without retraining. Our implementation successfully demonstrates inference-time control on MNIST with minimal supervision, despite limited compute resources. We suspect that trajectory-aware compute allocation—non-uniform distribution of inference effort across steps—can further enhance performance and generalization, and will investigate this in future work. Also, since we have generalized inference scaling of diffusion models in the stochastic domain, we intend to expand our work to domains that utilize stochastic diffusion models, like robotics and complex image and video generation.

The future_generalization_experiment directory contains a prototype to evaluate the discriminative power of the verifier across domains.

To the best of our knowledge, this is the first empirical and theoretical instantiation of trajectory space inference scaling for stochastic diffusion models.

We gratefully acknowledge compute support provided by researchers at DeepMind

-----------------------------------------------------------------------------------------------------------------------------------------------
Directory Configuration:
* main_experiments/ - Code and classifier models used for experiments in the paper.
* lc_gpu_accelerated_training/ - Code to train label-conditioned diffusion model for MNIST on a GPU. .env with lambda ssh key and api key required in main repo dir. 
* nlc_gpu_accelerated_training/ - Code to train non-label-conditioned diffusion model for MNIST on a GPU. .env with lambda ssh key and api key required in main repo dir. 
* inference_experiments/ - Production grade code with utils, refactored from main_experiments for folks who want more generalized code.
* model_tests/ - Basic model tests conducted in prepration for larger experiments. 
* future_generalization_experiment/ - Code for experiments we would like to perform on larger datasets and models.