# docker/mathlibs/mkl/Dockerfile
FROM scratch

# Install Intel MKL
RUN wget -q https://registrationcenter-download.intel.com/akdlm/irc_nas/tec/17415/l_dnnl_1.6.4.915.tar.gz -O oneDNN.tar.gz && \
    tar -xzf oneDNN.tar.gz && \
    cd l_dnnl_* && \
    ./install.sh && \
    cd .. && \
    rm -rf l_dnnl_*

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

