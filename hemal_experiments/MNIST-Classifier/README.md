# MNIST-Classifier
Pretrained classifier (Convolutional Neural Network, CNN) to classify MNIST images, based on Keras with the Tensorflow backend.

### Requirements:
* Keras 2.1.4
* Tensorflow 1.5.0
* Numpy 1.14.1

### To predict images:
To predict existing images with the pre-trained model (99.36% accuracy on the MNIST test set)
* `python mnist_classifier.py --predict --model weights.hdf5 --img_path path-to-images`

Images should be stored in the following layout:
* path-to-images
    * class-0
        * img1.jpg
        * img2.jpg
        * ...
    * class-1
        * img1.jpg
        * img2.jpg
        * ...
    * ...


### To train a new classifier
To train a new classifier on the MNIST data:
* `python mnist_classifier.py --train`

To view training statistics:
* `tensorboard --logdir log_dir/`

Check out command line arguments for further control over the hyperparameters used for training:
* `python mnist_classifier.py --help`

