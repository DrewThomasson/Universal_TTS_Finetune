#!/bin/bash
set -e

if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
python xtts_demo.py "$@"
