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

#### FROM NVIDIA DOCKER HUB REPO###########
ENV CUDA_VERSION=11.1.1

RUN apt-get update && apt-get install -y --no-install-recommends     gnupg2 curl ca-certificates &&     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list &&     echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list &&     apt-get purge --autoremove -y curl     && rm -rf /var/lib/apt/lists/* # buildkit

RUN apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-11-1=11.1.74-1     cuda-compat-11-1     && ln -s cuda-11.1 /usr/local/cuda &&     rm -rf /var/lib/apt/lists/* # buildkit

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit

ENV PATH=$PATH:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450

ENV NCCL_VERSION=2.8.4
RUN apt-get update && apt-get install -y --no-install-recommends     cuda-libraries-11-1=11.1.1-1     libnpp-11-1=11.1.2.301-1     cuda-nvtx-11-1=11.1.74-1     libcublas-11-1=11.3.0.106-1     libcusparse-11-1=11.3.0.10-1     libnccl2=$NCCL_VERSION-1+cuda11.1     && rm -rf /var/lib/apt/lists/* # buildkit

RUN apt-mark hold libcublas-11-1 libnccl2 # buildkit
ENV NCCL_VERSION=2.8.4
RUN apt-get update && apt-get install -y --no-install-recommends     libtinfo5 libncursesw5     cuda-cudart-dev-11-1=11.1.74-1     cuda-command-line-tools-11-1=11.1.1-1     cuda-minimal-build-11-1=11.1.1-1     cuda-libraries-dev-11-1=11.1.1-1     cuda-nvml-dev-11-1=11.1.74-1     libnpp-dev-11-1=11.1.2.301-1     libnccl-dev=2.8.4-1+cuda11.1     libcublas-dev-11-1=11.3.0.106-1     libcusparse-dev-11-1=11.3.0.10-1     && rm -rf /var/lib/apt/lists/* # buildkit

RUN apt-mark hold libcublas-dev-11-1 libnccl-dev # buildkit
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
########################################

RUN pip install tensorflow
# don't know why pip pytorch keep giving me connection reset.
RUN /opt/conda/bin/conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

WORKDIR /root/codes
RUN git clone https://github.com/matthewygf/automl.git
RUN git clone https://github.com/matthewygf/yolov5.git
ENV PYTHONPATH=$PYTHONPATH:/root/codes/automl

RUN pip install pandas numpy

RUN apt-get install ffmpeg libsm6 libxext6  -y

# clone automl
WORKDIR /root/codes

ENTRYPOINT [ "/bin/bash" ]