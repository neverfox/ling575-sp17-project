# Neural Dependency Parser

This repository contains a PyTorch implementation of a neural network architecture for multilingual syntactic dependency parsing in response to the CoNLL 2017 Shared Task and drawing from the work of [Kiperwasser and Goldberg (2016)](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198), [Dozat and Manning (2016)](https://arxiv.org/pdf/1611.01734.pdf), and [Ma and Hovy (2017)](https://arxiv.org/pdf/1701.00874.pdf). The architecture breaks the parsing problem into an encoder for transforming annotated text into a feature representation using bidirectional LSTMs, a biaffine structured attention mechanism for scoring potential labeled arcs, and a graph-based CRF for determining loss.

# Installation
The code was developed in OS X in an Anaconda 4.3.1 environment with Python 3. Assuming a similarly configured environment, install the dependencies using the included requirements files for both `conda` and `pip`. Additionally, you must install [torchtext](https://github.com/pytorch/text) manually by cloning the source and using `python setup.py install` from the base of that repository. It should also work in other *nix/Python setups, but may require some modifications.

Alternatively, if you have Docker, you can build an image that contains everything you need using the included Dockerfile. This Docker image can even work with a CUDA GPU if you use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

# Data
None of the data is included in this repository. You must download and extract the CoNLL 2017 Shared Task data in the `input` folder:

```bash
$ wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1983/ud-treebanks-conll2017.tgz?sequence=2&isAllowed=y | tar xvz -C ./input
$ wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2184/ud-test-v2.0-conll2017.tgz?sequence=3&isAllowed=y | tar xvz -C ./input
```

You will also want the [word embeddings](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y), which are over 22GB of data compressed. These so far need to be handled manually be extracting the relevant .xz file from the archive, removing the first line of text, and using `torch.save` to create a `<lang>.100d.pt` in the `input` directory, eg. `en.100d.pt`.

# Running
To run a model, execute `python main.py` from the `src` directory. For relevant command line options (there are quite a few), see `src/utils.py`. By default the code trains and tests an English model on the CPU with reasonable defaults. It is highly recommended, however, that you run the model on a CUDA-enabled GPU with the `--cuda` flag. The model will save checkouts after each epoch and a final model file to the `output` directory.

# Hyperparameter optimization
To optimize hyperparameters, run `python hyperband.py` which can take all of the same command line options as the main program. Feel free to tweak the exploration space located in that file. The results are saved as `output/<lang>.params.pt` and can be loaded later (with `torch.load`).

# Results
For an explanation of the architecture and some initial results using this code, see the included PDF article.
