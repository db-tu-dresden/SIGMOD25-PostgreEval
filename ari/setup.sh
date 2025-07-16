#!/bin/bash

docker build -t ari-elephant .

docker run -dt \
    --name ari-elephant \
    --volume $PWD/ari-elephant:/ari \
    --shm-size=256G \
    -e SETUP_JOB=true \
    ari-elephant
