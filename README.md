# Neural Dependency Parser

This repository contains a PyTorch implementation of a neural network architecture for multilingual syntactic dependency parsing in response to the CoNLL 2017 Shared Task and drawing from the work of [Kiperwasser and Goldberg (2016)](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198), [Dozat and Manning (2016)](https://arxiv.org/pdf/1611.01734.pdf), and [Ma and Hovy (2017)](https://arxiv.org/pdf/1701.00874.pdf). The architecture breaks the parsing problem into an encoder for transforming annotated text into a feature representation using bidirectional LSTMs, a biaffine structured attention mechanism for scoring potential labeled arcs, and a graph-based CRF for determining loss.

# Installation
To run the code requires an machine with a CUDA-enabled GPU with a Docker environment and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). You can build an image that contains everything else you need using the included Dockerfile.

```bash
$ docker build -t parser .
$ nvidia-docker run --rm -it --ipc=host -v "$PWD":/workspace parser /bin/bash
```

# Data
None of the data is included in this repository. You must download and extract the CoNLL 2017 Shared Task data in the `input` folder:

```bash
$ wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1983/ud-treebanks-conll2017.tgz?sequence=2&isAllowed=y | tar xvz -C ./input
$ wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2184/ud-test-v2.0-conll2017.tgz?sequence=3&isAllowed=y | tar xvz -C ./input
```

You will also want the [word embeddings](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y), which are over 22GB of data compressed. These so far need to be handled manually be extracting the relevant .xz file from the archive, removing the first line of text, and using `torch.save` to create a `<lang>.100d.pt` in the `input` directory, eg. `en.100d.pt`.

Currently, the code is configured to access data for English, Chinese, and Vietnamese. To add more language support, edit the lang dict in `conll_u.py` with the appropriate codes and match names using the downloaded data as a guide.

# Running
To run a model from inside the running Docker container, execute `python main.py` from the `src` directory. For relevant command line options (there are quite a few), see `src/utils.py`. By default the code trains and tests an English model on the CPU with reasonable defaults. It is *highly* recommended, however, that you run the model on a CUDA-enabled GPU with the `--cuda` flag. The model will save checkouts after each epoch and a final model file to the `output` directory.

# Hyperparameter optimization
To optimize hyperparameters, run `python hyperband.py` which can take all of the same command line options as the main program. Feel free to tweak the exploration space located in that file. The results are saved as `output/<lang>.params.pt` and can be loaded later (with `torch.load`).

# Results
For an explanation of the architecture and some initial results using this code, see the included PDF article.

English

```
Trained epoch 1 - Loss: 18.4695, Learning Rate: 0.001537
Validated epoch 1 - Loss: 9.4566, UAS: 85.983, LAS: 83.456
Trained epoch 2 - Loss: 11.3783, Learning Rate: 0.001477
Validated epoch 2 - Loss: 7.9728, UAS: 87.849, LAS: 85.569
Trained epoch 3 - Loss: 9.9963, Learning Rate: 0.001419
Validated epoch 3 - Loss: 7.5522, UAS: 89.035, LAS: 86.809
Trained epoch 4 - Loss: 9.1857, Learning Rate: 0.001364
Validated epoch 4 - Loss: 7.6052, UAS: 89.333, LAS: 87.140
Trained epoch 5 - Loss: 8.3636, Learning Rate: 0.001310
Validated epoch 5 - Loss: 7.2680, UAS: 89.163, LAS: 86.907
Trained epoch 6 - Loss: 7.8826, Learning Rate: 0.001259
Validated epoch 6 - Loss: 7.4391, UAS: 89.453, LAS: 87.174
Trained epoch 7 - Loss: 7.2777, Learning Rate: 0.001210
Validated epoch 7 - Loss: 7.2088, UAS: 89.464, LAS: 87.598
Trained epoch 8 - Loss: 6.8237, Learning Rate: 0.001163
Validated epoch 8 - Loss: 7.6121, UAS: 89.195, LAS: 87.260
Trained epoch 9 - Loss: 6.3731, Learning Rate: 0.001117
Validated epoch 9 - Loss: 7.9432, UAS: 89.598, LAS: 87.542
Trained epoch 10 - Loss: 6.0028, Learning Rate: 0.001073
Validated epoch 10 - Loss: 8.0941, UAS: 89.438, LAS: 87.484
Trained epoch 11 - Loss: 5.5719, Learning Rate: 0.001031
Validated epoch 11 - Loss: 8.3746, UAS: 89.780, LAS: 87.839
Trained epoch 12 - Loss: 5.2032, Learning Rate: 0.000991
Validated epoch 12 - Loss: 8.6846, UAS: 89.839, LAS: 87.902
Trained epoch 13 - Loss: 4.8610, Learning Rate: 0.000952
Validated epoch 13 - Loss: 8.5476, UAS: 89.890, LAS: 87.819
Trained epoch 14 - Loss: 4.5282, Learning Rate: 0.000915
Validated epoch 14 - Loss: 9.1374, UAS: 89.897, LAS: 87.877
Trained epoch 15 - Loss: 4.2962, Learning Rate: 0.000879
Validated epoch 15 - Loss: 9.0678, UAS: 89.407, LAS: 87.431
Trained epoch 16 - Loss: 3.9930, Learning Rate: 0.000845
Validated epoch 16 - Loss: 9.3954, UAS: 89.852, LAS: 87.865
Trained epoch 17 - Loss: 3.7993, Learning Rate: 0.000812
Validated epoch 17 - Loss: 9.5943, UAS: 89.802, LAS: 87.773

Tested model - Loss: 8.9735, UAS: 88.280, LAS: 86.365
```

Chinese

```
Trained epoch 1 - Loss: 45.2063, Learning Rate: 0.001537
Validated epoch 1 - Loss: 27.1977, UAS: 77.464, LAS: 73.466
Trained epoch 2 - Loss: 23.3192, Learning Rate: 0.001477
Validated epoch 2 - Loss: 23.5167, UAS: 80.776, LAS: 77.361
Trained epoch 3 - Loss: 19.3002, Learning Rate: 0.001419
Validated epoch 3 - Loss: 21.7034, UAS: 82.375, LAS: 78.874
Trained epoch 4 - Loss: 16.9629, Learning Rate: 0.001364
Validated epoch 4 - Loss: 21.4906, UAS: 82.549, LAS: 79.616
Trained epoch 5 - Loss: 14.9779, Learning Rate: 0.001310
Validated epoch 5 - Loss: 20.8838, UAS: 82.403, LAS: 79.531
Trained epoch 6 - Loss: 13.4654, Learning Rate: 0.001259
Validated epoch 6 - Loss: 20.8933, UAS: 82.735, LAS: 79.826
Trained epoch 7 - Loss: 12.0011, Learning Rate: 0.001210
Validated epoch 7 - Loss: 20.6931, UAS: 83.619, LAS: 80.739
Trained epoch 8 - Loss: 10.5491, Learning Rate: 0.001163
Validated epoch 8 - Loss: 22.6324, UAS: 82.696, LAS: 79.992
Trained epoch 9 - Loss: 9.3072, Learning Rate: 0.001117
Validated epoch 9 - Loss: 24.3105, UAS: 82.294, LAS: 79.528
Trained epoch 10 - Loss: 8.2510, Learning Rate: 0.001073
Validated epoch 10 - Loss: 25.9691, UAS: 82.493, LAS: 79.644
Trained epoch 11 - Loss: 7.3604, Learning Rate: 0.001031
Validated epoch 11 - Loss: 26.8273, UAS: 82.778, LAS: 79.907
Trained epoch 12 - Loss: 6.5419, Learning Rate: 0.000991
Validated epoch 12 - Loss: 26.9238, UAS: 82.721, LAS: 79.740

Tested model - Loss: 16.7507, UAS: 86.073, LAS: 83.513
```

Vietnamese

```
Trained epoch 1 - Loss: 33.7578, Learning Rate: 0.001537
Validated epoch 1 - Loss: 21.0471, UAS: 62.422, LAS: 59.097
Trained epoch 2 - Loss: 17.2494, Learning Rate: 0.001477
Validated epoch 2 - Loss: 18.3685, UAS: 68.175, LAS: 64.935
Trained epoch 3 - Loss: 13.3789, Learning Rate: 0.001419
Validated epoch 3 - Loss: 17.3818, UAS: 70.391, LAS: 67.598
Trained epoch 4 - Loss: 10.7135, Learning Rate: 0.001364
Validated epoch 4 - Loss: 17.8208, UAS: 70.173, LAS: 66.615
Trained epoch 5 - Loss: 8.9030, Learning Rate: 0.001310
Validated epoch 5 - Loss: 18.6925, UAS: 71.279, LAS: 68.310
Trained epoch 6 - Loss: 7.2058, Learning Rate: 0.001259
Validated epoch 6 - Loss: 21.3108, UAS: 70.939, LAS: 67.907
Trained epoch 7 - Loss: 5.8223, Learning Rate: 0.001210
Validated epoch 7 - Loss: 20.0631, UAS: 71.854, LAS: 68.742
Trained epoch 8 - Loss: 4.5989, Learning Rate: 0.001163
Validated epoch 8 - Loss: 22.9194, UAS: 72.034, LAS: 68.959
Trained epoch 9 - Loss: 3.8819, Learning Rate: 0.001117
Validated epoch 9 - Loss: 23.6619, UAS: 72.274, LAS: 68.560
Trained epoch 10 - Loss: 3.2770, Learning Rate: 0.001073
Validated epoch 10 - Loss: 25.8729, UAS: 72.060, LAS: 68.736
Trained epoch 11 - Loss: 2.7938, Learning Rate: 0.001031
Validated epoch 11 - Loss: 27.2023, UAS: 71.882, LAS: 68.804
Trained epoch 12 - Loss: 2.3240, Learning Rate: 0.000991
Validated epoch 12 - Loss: 28.7678, UAS: 71.610, LAS: 68.443
Trained epoch 13 - Loss: 1.8794, Learning Rate: 0.000952
Validated epoch 13 - Loss: 29.5592, UAS: 72.486, LAS: 69.244
Trained epoch 14 - Loss: 1.7114, Learning Rate: 0.000915
Validated epoch 14 - Loss: 33.0395, UAS: 72.494, LAS: 69.064
Trained epoch 15 - Loss: 1.5476, Learning Rate: 0.000879
Validated epoch 15 - Loss: 34.8316, UAS: 72.108, LAS: 68.436
Trained epoch 16 - Loss: 1.3774, Learning Rate: 0.000845
Validated epoch 16 - Loss: 33.7205, UAS: 71.625, LAS: 68.127
Trained epoch 17 - Loss: 1.2727, Learning Rate: 0.000812
Validated epoch 17 - Loss: 33.9114, UAS: 71.416, LAS: 67.996
Trained epoch 18 - Loss: 1.2094, Learning Rate: 0.000780
Validated epoch 18 - Loss: 35.8566, UAS: 71.747, LAS: 68.418

Tested model - Loss: 31.7525, UAS: 72.111, LAS: 69.088
```
