export TRAINING_DATA=input/trainSet1N1Balanced.csv # BEFORE FOLDS
# export TEST_DATA=input/test.csv

export MODEL=$1

python -m src.gridSearch
