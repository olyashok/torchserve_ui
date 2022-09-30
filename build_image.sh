#!/bin/bash

BASE_IMAGE="python:3.8"
DOCKER_TAG="streamlit/streamlit"

DOCKER_BUILDKIT=1 docker build --no-cache --file Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE -t $DOCKER_TAG .
