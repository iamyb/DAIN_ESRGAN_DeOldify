FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


# Install Miniconda in /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH /opt/conda/envs/ded/bin;$PATH


WORKDIR /root/workspace/
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda deactivate" >> ~/.bashrc && \
    echo "conda activate ded" >> ~/.bashrc

RUN apt-get update && apt-get install -y ffmpeg
RUN mkdir -p /root/.torch/models && wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O /root/.torch/models/resnet101-5d3b4d8f.pth
RUN wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth -O best.pth
RUN wget https://www.dropbox.com/s/336vn9y4qwyg9yz/ColorizeVideo_gen.pth?dl=0 -O ColorizeVideo_gen.pth
RUN wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene' -O RRDB_ESRGAN_x4.pth

ARG CONDA_DEFAULT_ENV=ded
ENV CONDA_DEFAULT_ENV ded
COPY . .

RUN /bin/bash -c "cd DAIN && git apply ../patches/dain.patch && cd .."
RUN /bin/bash -c "cd DeOldify && git apply ../patches/deoldify.patch && cd .."

#RUN /bin/bash -c "source activate ded && cd DAIN/my_package && ./build.sh && cd ../.."
#RUN /bin/bash -c "source activate ded && cd DAIN/PWCNet/correlation_package_pytorch1_0 && ./build.sh && cd ../../.."

RUN /bin/bash -c "mkdir -p DAIN/model_weights && mv best.pth DAIN/model_weights"

RUN /bin/bash -c "source activate ded && cd DAIN/my_package && ./build.sh && cd ../.."
RUN /bin/bash -c "source activate ded && cd DAIN/PWCNet/correlation_package_pytorch1_0 && ./build.sh && cd ../../.."

RUN /bin/bash -c "mkdir -p ESRGAN/models && mv RRDB_ESRGAN_x4.pth ESRGAN/models"
RUN /bin/bash -c "mkdir -p DeOldify/models && mv ColorizeVideo_gen.pth DeOldify/models"

