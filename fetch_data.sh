#!/bin/bash

if [ "$1" == "mush" ]; then
    echo -e "Fetching & Preprocessing Mushroom Data..."
    mkdir -p data/
    wget -P data/ https://archive.ics.uci.edu/static/public/73/mushroom.zip
    cd data/
    unzip mushroom.zip && rm mushroom.zip
    mv agaricus-lepiota.data mushrooms.csv
    ls | grep -v .csv | xargs rm
    cd ..
    python src/Mushrooms.py
    rm data/mushrooms.csv
    echo -e "Mushroom data fetched and preprocessed successfully!"
elif [ "$1" == "rcv1" ]; then
    echo -e "Fetching & Preprocessing RCV1 Data..."
    mkdir -p data/
    python src/RCV1.py
    echo -e "Mushroom data fetched and preprocessed successfully!"
else
    echo -e "Error: Unavailable dataset. Please provide 'mush' or 'rcv1' as an argument."
    exit 1
fi