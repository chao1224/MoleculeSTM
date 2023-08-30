FROM nvcr.io/nvidia/pytorch:22.01-py3 as base

#create a new new user
RUN useradd -ms /bin/bash shengchaol

# #change to this user
# USER shengchaol

#set working directory
WORKDIR /home/shengchaol

RUN chmod -R 777 /home/shengchaol
RUN chmod -R 777 /usr/bin
RUN chmod -R 777 /bin
RUN chmod -R 777 /usr/local
RUN chmod -R 777 /opt/conda

RUN conda install -y python=3.7

RUN conda install -y -c rdkit rdkit=2020.09.1.0
RUN conda install -y -c conda-forge -c pytorch pytorch=1.9.1

RUN conda install -y -c pyg -c conda-forge pyg

RUN pip install requests
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install spacy

# for SciBert
RUN conda install -y boto3
RUN pip install transformers

# for MoleculeNet
RUN pip install ogb

# install pysmilesutils
RUN python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git

RUN pip install deepspeed

# install Megatron
RUN cd /tmp && git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc && cd /tmp/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism && pip install .

# install apex
RUN cd /tmp && git clone https://github.com/chao1224/apex.git
RUN cd /tmp/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


#expose port for Jupyter
EXPOSE 8888