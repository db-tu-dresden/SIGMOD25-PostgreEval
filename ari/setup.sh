#!/bin/bash

docker build -t ari-elephant .

docker run -dt \
    --name ari-elephant \
    --volume $PWD/docker-volume:/ari \
    --publish 8888:8888 \
    --shm-size=256G \
    -e SETUP_JOB=true \
    ari-elephant
