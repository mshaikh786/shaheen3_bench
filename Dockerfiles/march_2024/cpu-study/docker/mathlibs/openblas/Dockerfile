# docker/mathlibs/openblas/Dockerfile
FROM scratch

RUN apt-get update && \
    apt-get install -y --no-install-recommends libopenblas-dev

ENV OPENBLASROOT=/usr
ENV LD_LIBRARY_PATH=$OPENBLASROOT/lib:$LD_LIBRARY_PATH

