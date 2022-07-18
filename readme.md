# Addressing Class Imbalance in Neuropathology through Generative Adversarial Networks

---

Author: Ciaran Coleman (ciaran.coleman6@gmail.com)

This repository contains the code and report used for my Master's thesis at UCL.

PlaqueGAN is an unconditional GAN for synthesising Amyloid beta plaques in immunohistochemically stained slides of human brain tissue. PlaqueGAN was specifically conceived to generate minority plaques (cored, CAA) - which are important to diagnosis of neurodegenerative diseases - to bolster datasets skewed with a majority of diffuse plaques (not so important to disease diagnosis). At the time of writing, this presented a first in generative modelling of these morphologies.

## Image data

---

This work would not have been possible without the datasets put together by [Tang et al][1] and [Wong et al][2]. Please refer to their repositories for access to the full datasets and description of how the tiles were processed.

Only a small subset of the Tang et al dataset, focussing on tiles that are labelled with minority plaques, were used for PlaqueGAN training. These have been stored in hdf5 format for file management ease and are available at: <https://osf.io/fc8pu/>

Please download the four hdf5 files to the <code>train_data</code> folder in the repository. See [here](#recommended-directory-structure) for more details.

[1]: <https://github.com/keiserlab/plaquebox-paper> "Plaquebox paper"

[2]: <https://github.com/keiserlab/consensus-learning-paper> "Consensus learning paper"

## Python packages required

---

Below are the packages and relevant versions tested. While some additional testing has since been done in Google Colab, there is no guarantee other/ later versions of these packages will be compatible. We therefore recommend using Anaconda3 to create a virtual environment with these as a starting point. If you wish only to train PlaqueGAN and calculate metrics, a smaller set of packages will suffice.

**To train PlaqueGAN:**
```
python            3.9.6         h6244533_1
numpy             1.20.2        py39ha4e8547_0
pandas            1.2.4         py39hd77b12b_0
scikit-learn      0.24.2        py39hf11a4ad_1
scikit-image      0.18.1        py39hf11a4ad_0
scipy             1.6.2         py39h66253e8_1
pytorch           1.9.0         py3.9_cuda11.1_cudnn8_0 (pytorch)
torchvision       0.2.2         py_3 (pytorch)
kornia            0.5.8         pyhd8ed1ab_0 (conda-forge)
tqdm              4.59.0        pyhd3eb1b0_1        
pillow            8.3.1         py39h4fa10fc_0
h5py              3.2.1         py39h3de5c98_0
einops            0.3.0         py_0 (conda-forge)
pytorch-fid       0.2.0         pypi_0 (pypi)
```
**To run other code/ notebooks:**
```
ipython           7.26.0        py39hd4e2768_0
jupyter           1.0.0         py39haa95532_7
matplotlib        3.3.4         py39haa95532_0
umap-learn        0.5.1         py39hcbf5309_1 (conda-forge)
imbalanced-learn  0.8.0         pyhd8ed1ab_0 (conda-forge)   
```
## System requirements

---

PlaqueGAN was developed with the aim of being widely accessible. All code was developed on a modest laptop with Windows 10 OS and is yet untested on Linux or MacOS.

- Deep learning models were trained using a single NVIDIA GeForce GTX-1060 GPU with 6GB VRAM and 1280 CUDA cores.
- 12 core Intel Core i7-8750H CPU
- 16 GB RAM

Training PlaqueGAN to 100,000 iterations takes approximately 24hrs with this setup. Faster times can easily be achieved with more powerful GPUs (e.g. in Google Colab) but requires editing of the code to work with multi-GPU setups.

## Recommended directory structure

---

For minimal editing of the codebase, it is recommended that the repository is downloaded as is, and that the training data be downloaded into the <code>train_data</code> folder as follows:

```
|--PlaqueGAN
|  |--train_data
|     |--caa-diffuse_train.h5
|     |--caa_train.h5
|     |--cored-diffuse_train.h5
|     |--cored_train.h5
|   |--lpips
|  |--metrics
|--train.py
|--etc.
```

## Training PlaqueGAN

---

To train PlaqueGAN, please download/clone/fork this repo. Change directory to the PlaqueGAN folder and activate the virtual environment (if using). The following can then be run as an example:

<code>!python train.py --path=./train_data/cored_train.h5 --name=cored --model_config=plaquegan --train_config=plaquegan --iter=100000 --with_amp=0 --batch_size=8 --im_size=256 --log_every=10</code>

- <code>path</code>: Path to the hdf5 training data.
- <code>name</code>: Desired name of the experiment to store training progress in.
- <code>model_config</code>: Which of the model configurations in model_configs.csv to run. <code>plaquegan</code> is the default.
- <code>train_config</code>: Which of the training configurations in training_configs.csv to run. <code>plaquegan</code> is the default.
-<code>iter</code>: The number of iterations to run for training.
-<code>with_amp</code>: Whether to run Automatic Mixed Precision training. It is recommended to only use this when the model is unable to fit into GPU memory.
-<code>batch_size</code>: batch size of images. Default is <code>8</code>.
-<code>im_size</code>: Size in pixels of width/height of images to be synthesised. Currently, code is only adapted to square images. Default is <code>256</code> for these datasets.
-<code>log_every</code>: How often to log losses etc. during training. This **does not** include frequency of images sampled from the Generator during training.

### Monitoring training progress

A results folder <code>train_results</code> is automatically created if it does not already exist. Within, a subfolder depending on <code>name</code> is also created- this is where the training progress is stored for the particular experiment. A snapshot of the scripts used to run PlaqueGAN are also transferred to this folder for traceability.

The structure of the subfolder is as so:

```
|--name
|  |--evaluation
|  |--images
|  |--models
|--args.txt
|--train.py
|--etc.
```
- <code>args.txt</code>: dictionary of all arguments used to train PlaqueGAN. This is for traceability but also used to reconstruct the Generator/ Discriminator when loading in models to evaluate.  
- <code>evaluation</code> directory is currently not created during training, but should be where metrics are stored if calculated while PlaqueGAN trains. For now, it is created when evaluating PlaqueGAN after training - see [this section](#calculating-quantitative-metrics) for more information.
- <code>images</code> directory consists of:
  - 4x2 grids of images sampled from the Generator every 1000 iterations. These are labelled <code>0.jpg</code>, <code>1000.jpg</code>, <code>2000.jpg</code> etc. This uses a fixed batch of noise vectors sampled at the beginning of training so that it is easier to review how training is progressing.
  - 8x3 grid of images, also sampled every 1000 iterations:
    - the top row is a random sample of real training images, with differential augmentations applied.
    - the middle row is the PlaqueGAN main Discriminator's reconstruction of the top row from the <code>8 x 8</code> feature maps.
    - the bottom row is the PlaqueGAN main Discriminator's reconstruction of the cropped version of the top row from the <code>16 x 16</code> feature maps.
- <code>models</code> directory contains snapshots of PlaqueGAN (in .pth format) at 5000 iteration intervals of training. The versions with <code>all_</code> at the beginning of the filename contain all necessary information to continue training. The lightweight versions only contain state dictionaries for the Generator and Discriminators.


### Continuing training from a checkpoint

If you wish to continue a training run, follow the format of the command line below, with the path to the full model. The example shows how to continue training the 'cored' experiment from 100,000 iterations to 150,000 iterations:

<code>!python train.py --path=./train_data/cored_train.h5 --name=cored --model_config=plaquegan --train_config=plaquegan --iter=150000 --ckpt=./train_results/cored/models/all_100000.pth</code>


## Analysing results

---

Once PlaqueGAN is trained, you may wish to generate samples or evaluate them both quantitatively and qualitatively. This section provides some ideas on how to do so, but you may have to go digging into the jupyter notebooks as specific scripts have not yet been written for them.

### Generating samples

To generate samples, our recommendation is to use the notebook <code>plot_image_grids.ipynb</code> for examples and then write a python script specific if it will be repeated frequently.

<code>generate_samples.ipynb</code> may also be useful but please note that you will also need to load in the trained model from the Tang et al repo if the intention is to filter out 'poor' samples using class confidence from a pre-trained CNN classifier.

#### An important note on using <code>eval()</code> mode to sample

You may notice in this code that, when generating samples, PlaqueGAN 's Generator is not switched to <code>eval()</code> mode.

PlaqueGAN makes use of batch normalisation in its layers - when placed in <code>eval()</code> mode, it reverts to running average mean and variance statistics seen during training. When using the exponential moving averages of the Generator's weights for sampling, the batch norm running averages are poor estimates as they aren't updated in such a way. This leads to greatly degraded image quality.

To counteract this, we leave the model in <code>train()</code> mode and instead *warm up* the Generator by running through 100 forward passes with different batches of random noise (discarding the generated samples). Only after this warm up are samples stored.

**N.B.:** It might be better to switch the model to <code>eval()</code> after the warm up is complete to give a standing statistic as described in Brock et al's BigGAN paper. I leave this to the reader to experiment with, as well as removing batch normalization layers altogether as is more common in recent GANs!

### Calculating quantitative metrics

There are two scripts used to evaluate PlaqueGAN quantitatively.

<code>kid_experiment.py</code> is used to calculate the Kernel Inception Distance. To run the script, use the following as an example:

<code>!python kid_experiment.py --path=./train_results --data_path=./train_data/cored_train.h5 --name=cored --start_iter=5000 --end_iter=150000 --feature_size=2048</code>

For a full set of arguments please refer to the script.
- <code>path</code>: main folder where experiment results are stored. Here, it is the <code>train_results</code> directory.
- <code>data_path</code>: path to the hdf5 file of *real* image data you wish to compare to.
- <code>start_iter</code>: which iteration of model training you wish to start at.
- <code>end_iter</code>: which iteration of model training you wish to end at.
- <code>feature_size</code>: which feature size to use in the inception network. Default is <code>2048</code>.


<code>prdc_nn_experiment.py</code> calculates Precision, Recall, Density & Coverage metrics, as well as the leave one out 1-nearest neighbours accuracies. Running the script requires a similar format to the one above:

<code>!python prdc_nn_experiment.py --path=./train_results --data_path=./train_data/cored_train.h5 --name=cored --start_iter=5000 --end_iter=150000 --embedding_type=T4096 --n_k=5</code>

For a full set of arguments please refer to the script.
- <code>embedding_type</code>: whether to used trained or random VGG16 embeddings. Default is trained <code>T4096</code> embeddings.
- <code>n_k</code>: How many nearest neighbours to use for calculation of PRDC.

### Evaluating visually

Please see the jupyter notebook <code>plot_image_grids.ipynb</code> for examples on how to do nearest neighbours and spherical interpolation between two generated samples.

## Future plans

---

- Add description of how to run on Google Colab.
- Build docker container for code.

## Other Acknowledgements

---

- Original [FastGAN][3] PyTorch code by Liu et al and [third-party][4] implementation by Phil Wang.

- [KID][5] by Mikolaj Binkowski et al.

- [PRDC][6] by Oh et al.

- [LOO 1-NN accuracies][7] by Xu et al.

[3]: <https://github.com/odegeasslbc/FastGAN-pytorch> "FastGAN"

[4]: <https://github.com/lucidrains/lightweight-gan> "FastGAN alternative"

[5]: <https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py> "KID"

[6]: <https://github.com/clovaai/generative-evaluation-prdc> "PRDC"

[7]: <https://github.com/xuqiantong/GAN-Metrics> "LOO 1-NN"
