#!/bin/sh

container=torchserve_ui

docker stop $container
docker rm $container
docker run -ti -d --runtime=nvidia --ipc=host -p 8502:8502 --net="host" --restart=always -v /mnt/localshared/data/hassio:/mnt/localshared/data/hassio -v /mnt/nas_downloads/deepstack/streamlit/streamlit:/streamlit -v /mnt/nas_downloads:/mnt/nas_downloads -v torchserve_model_store:/home/model-server/model-store --name $container streamlit/streamlit:latest
