#!/bin/bash

> benchmark.csv

echo "config, energy_time, min_cost_time, total_time" >> benchmark.csv

echo -n "small-0-0-smooth, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-0-0-smooth.jpg -w 84 -h 1000 -e 0 -m 0 -s 2>> benchmark.csv
echo -n "medium-0-0-smooth, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-0-smooth.jpg -w 317 -h 1000 -e 0 -m 0 -s 2>> benchmark.csv
echo -n "large-0-0-smooth, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-0-0-smooth.jpg -w 610 -h 1000 -e 0 -m 0 -s 2>> benchmark.csv

echo -n "small-1-1-smooth, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-1-1-smooth.jpg -w 84 -h 1000 -e 1 -m 1 -s 2>> benchmark.csv
echo -n "medium-1-1-smooth, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-1-1-smooth.jpg -w 317 -h 1000 -e 1 -m 1 -s 2>> benchmark.csv
echo -n "large-1-1-smooth, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-1-1-smooth.jpg -w 610 -h 1000 -e 1 -m 1 -s 2>> benchmark.csv

echo -n "small-2-2-smooth, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-2-2-smooth.jpg -w 84 -h 1000 -e 2 -m 2 -s 2>> benchmark.csv
echo -n "medium-2-2-smooth, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-2-2-smooth.jpg -w 317 -h 1000 -e 2 -m 2 -s 2>> benchmark.csv
echo -n "large-2-2-smooth, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-2-2-smooth.jpg -w 610 -h 1000 -e 2 -m 2 -s 2>> benchmark.csv

echo -n "small-0-0-rough, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-0-0-rough.jpg -w 84 -h 1000 -e 0 -m 0 2>> benchmark.csv
echo -n "medium-0-0-rough, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-0-0-rough.jpg -w 317 -h 1000 -e 0 -m 0 2>> benchmark.csv
echo -n "large-0-0-rough, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-0-0-rough.jpg -w 610 -h 1000 -e 0 -m 0 2>> benchmark.csv

echo -n "small-1-1-rough, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-1-1-rough.jpg -w 84 -h 1000 -e 1 -m 1 2>> benchmark.csv
echo -n "medium-1-1-rough, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-1-1-rough.jpg -w 317 -h 1000 -e 1 -m 1 2>> benchmark.csv
echo -n "large-1-1-rough, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-1-1-rough.jpg -w 610 -h 455 -e 1 -m 1 2>> benchmark.csv

echo -n "small-2-2-rough, " >> benchmark.csv
./seamcarver -i "input/small.jpg" -o output/small-2-2-rough.jpg -w 84 -h 1000 -e 2 -m 2 2>> benchmark.csv
echo -n "medium-2-2-rough, " >> benchmark.csv
./seamcarver -i "input/medium.jpg" -o output/medium-2-2-rough.jpg -w 317 -h 1000 -e 2 -m 2 2>> benchmark.csv
echo -n "large-2-2-rough, " >> benchmark.csv
./seamcarver -i "input/large.jpg" -o output/large-2-2-rough.jpg -w 610 -h 1000 -e 2 -m 2 2>> benchmark.csv
