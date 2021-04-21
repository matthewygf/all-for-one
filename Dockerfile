FROM ubuntu:20.04


ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install build-essential wget curl -y

RUN wget --quiet \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O ~/miniconda.sh &&     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh &&     /opt/conda/bin/conda clean -tipsy # buildkit

ARG PYTHON_VERSION=3.8
RUN PYTHON_VERSION=3.8 /opt/conda/bin/conda install --update-all -y -c defaults \
     conda conda-package-handling     python=$PYTHON_VERSION pip pycosat requests ruamel_yaml cytoolz  \
        anaconda-client nbformat make     pytest pytest-cov codecov radon pytest-timeout mock responses \
        pexpect     flake8 &&     /opt/conda/bin/conda clean --all --yes # buildkit

RUN PYTHON_VERSION=3.8 /opt/conda/bin/conda install --update-all -y -c defaults  \
       conda-build patch git         perl pytest-xdist pytest-mock         anaconda-client      \
          filelock jinja2 conda-verify pkginfo         glob2 beautifulsoup4 chardet pycrypto   \
            && /opt/conda/bin/conda clean --all --yes # buildkit


RUN pip install tensorflow
# don't know why pip pytorch keep giving me connection reset.
RUN /opt/conda/bin/conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

WORKDIR /root/codes
RUN git clone https://github.com/matthewygf/automl.git
ENV PYTHONPATH=$PYTHONPATH:/root/codes/automl

# clone automl
WORKDIR /root/codes