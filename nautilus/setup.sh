#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
export PATH="/root/.local/bin:$PATH"
micromamba run -n poetry pipx install poetry

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n acc python=3.11 -c conda-forge -y
export PATH="/root/.local/bin:$PATH"
micromamba run -n acc apt-get update -y 
micromamba run -n acc apt-get install ffmpeg libsm6 libxext6  -y
micromamba run -n acc apt-get install gcc -y
micromamba run -n acc poetry install
micromamba run -n acc poetry run inv setup
