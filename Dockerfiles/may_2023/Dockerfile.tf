FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3 AS builder
WORKDIR /software
#RUN git clone https://github.com/SilvioGiancola/SoccerNetv2-DevKit.git \
RUN  pip install  --prefix=/software SoccerNet==0.1.8 opencv-python==3.4.11.45 imutils moviepy \
&&  mkdir /data
ENV PYTHONPATH=/software/lib/python3.8/site-packages:${PYTHONPATH}
ENV SOCCERNET_INSTALL_DIR=/software
COPY download_data.py /data/download_data.py
