@echo off

if exist venv\scripts\activate (
  call venv\scripts\activate
)
python xtts_demo.py %*
