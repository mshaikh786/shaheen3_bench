# docker/mathlibs/blis/Dockerfile
FROM scratch

# Install BLIS
RUN git clone https://github.com/flame/blis /blis && \
    cd /blis && \
    ./configure auto && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf /blis

ENV BLISROOT=/usr/local
ENV LD_LIBRARY_PATH=$BLISROOT/lib:$LD_LIBRARY_PATH

