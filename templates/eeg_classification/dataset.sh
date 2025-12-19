#!/bin/bash

mkdir -p data

for subj in A01 A02 A03 A04 A05 A06 A07 A08 A09
do
    for sess in T E
    do
        fname="${subj}${sess}.mat"
        url="https://bnci-horizon-2020.eu/database/data-sets/001-2014/$fname"
        wget -O "./data/$fname" "$url"
    done
done