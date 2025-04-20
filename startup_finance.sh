#!/bin/bash
source /home/abhishek/anaconda3/bin/activate
conda activate finance-analysis
cd /home/abhishek/Desktop/Projects/finances
python app.py &
python -m src.run --schedule &