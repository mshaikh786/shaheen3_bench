FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3 AS builder
#FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3 AS builder
WORKDIR /software
#RUN git clone https://github.com/SilvioGiancola/SoccerNetv2-DevKit.git \
RUN  pip install -v --prefix=/software --verbose SoccerNet==0.1.8 
RUN  pip install -v --prefix=/software --verbose opencv-python
#==3.4.11.45
RUN  pip install -v --prefix=/software --verbose imutils 
RUN  pip install -v --prefix=/software --verbose moviepy 
RUN  pip install -v --prefix=/software --verbose torch==1.13.1
RUN  mkdir /data
ENV PYTHONPATH=/software/lib/python3.8/site-packages:${PYTHONPATH}
ENV SOCCERNET_INSTALL_DIR=/software
COPY download_data.py /data/download_data.py
