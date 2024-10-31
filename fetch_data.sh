#!/bin/bash
echo -e "Fetching and preprocessing data..."
mkdir -p data/
wget -P data/ https://archive.ics.uci.edu/static/public/73/mushroom.zip
cd data/
unzip mushroom.zip && rm mushroom.zip
mv agaricus-lepiota.data mushrooms.csv
ls | grep -v .csv | xargs rm
cd ..
python preprocess_data.py
rm data/mushrooms.csv
echo -e "Data fetched and preprocessed successfully!"