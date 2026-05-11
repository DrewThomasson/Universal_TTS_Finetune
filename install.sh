#!/bin/bash
set -e

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
