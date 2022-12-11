# Simulation of Traffic control (SUMO) for Multi-Agent Reinforcement Learning (MARL)


This codes has only been confirmed on Windows.
## Create Virual Env
python -m venv env

## Activate Virual Env
source /path/env/Scripts/activate.bat

## Install SUMO
Installing SUMO from [website](https://sumo.dlr.de/docs/Downloads.php) creates SUMO files under "C:\Program Files (x86)\Eclipse\Sumo".

## Install Requirmetns packages
pip install -r .\requirements.txt

## Change the Configuration for your network 
change files in sumo.cfg file
change the parameters in the src python code files

## Run the code
python main.py

