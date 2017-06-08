FROM pytorch-cudnnv6

RUN cd /tmp && git clone https://github.com/pytorch/text.git && cd text && \
    python setup.py install && cd .. && rm -rf text && \
    conda install --name pytorch-py35 -c conda-forge nltk tqdm && \
    conda install --name pytorch-py35 -c jaikumarm hyperopt && \
    pip install conllu dropbox
