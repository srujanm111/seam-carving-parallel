#!/bin/bash

> benchmark-energy.csv

echo "config, energy_time, min_cost_time, total_time" >> benchmark-energy.csv

echo -n "0-0-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-0-0.jpg -w 50 -h 50 -e 0 -m 0
echo -n "0-1-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-0-1.jpg -w 50 -h 50 -e 0 -m 1
echo -n "0-2-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-0-2.jpg -w 50 -h 50 -e 0 -m 2
echo -n "0-3-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-0-3.jpg -w 50 -h 50 -e 0 -m 3

echo -n "0-0-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-0.jpg -w 150 -h 150 -e 0 -m 0
echo -n "0-1-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-1.jpg -w 150 -h 150 -e 0 -m 1
echo -n "0-2-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-2.jpg -w 150 -h 150 -e 0 -m 2
echo -n "0-3-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-3.jpg -w 150 -h 150 -e 0 -m 3

echo -n "0-0-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-0-0.jpg -w 500 -h 500 -e 0 -m 0
echo -n "0-1-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-0-1.jpg -w 500 -h 500 -e 0 -m 1
echo -n "0-2-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-0-2.jpg -w 500 -h 500 -e 0 -m 2
echo -n "0-3-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-0-3.jpg -w 500 -h 500 -e 0 -m 3

> benchmark-mincost.csv

echo "config, energy_time, min_cost_time, total_time" >> benchmark-mincost.csv

echo -n "0-0-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-0-0.jpg -w 50 -h 50 -e 0 -m 0
echo -n "1-0-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-1-0.jpg -w 50 -h 50 -e 1 -m 0
echo -n "2-0-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-2-0.jpg -w 50 -h 50 -e 2 -m 0
echo -n "3-0-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-3-0.jpg -w 50 -h 50 -e 3 -m 0

echo -n "0-0-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-0.jpg -w 150 -h 150 -e 0 -m 0
echo -n "1-0-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-1-0.jpg -w 150 -h 150 -e 1 -m 0
echo -n "2-0-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-2-0.jpg -w 150 -h 150 -e 2 -m 0
echo -n "3-0-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-3-0.jpg -w 150 -h 150 -e 3 -m 0

echo -n "0-0-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-0-0.jpg -w 500 -h 500 -e 0 -m 0
echo -n "1-0-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-1-0.jpg -w 500 -h 500 -e 1 -m 0
echo -n "2-0-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-2-0.jpg -w 500 -h 500 -e 2 -m 0
echo -n "3-0-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-3-0.jpg -w 500 -h 500 -e 3 -m 0

> benchmark-total.csv

echo "config, energy_time, min_cost_time, total_time" >> benchmark-total.csv

echo -n "smooth-small, " >> benchmark-energy.csv
./seamcarver -i "input/small.jpg" -o output/small-smooth.jpg -w 50 -h 50 -e 0 -m 0 -s
echo -n "smooth-medium, " >> benchmark-energy.csv
./seamcarver -i "input/medium.jpg" -o output/medium-smooth.jpg -w 150 -h 150 -e 0 -m 0 -s
echo -n "smooth-large, " >> benchmark-energy.csv
./seamcarver -i "input/large.jpg" -o output/large-smooth.jpg -w 500 -h 500 -e 0 -m 0 -s
