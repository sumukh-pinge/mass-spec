#!/bin/bash

# ./main [TRAIN dataset path] [DIM] [ITER] [Quantize] [K]
echo  "Atom"
./main FCPS/Atom.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Atom.csv 10000 10 256 8
python3 result_parser.py

echo "Chainlink"
./main FCPS/Chainlink.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Chainlink.csv 10000 10 256 8
python3 result_parser.py

echo "EngyTime"
./main FCPS/EngyTime.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/EngyTime.csv 10000 10 256 8
python3 result_parser.py

echo "Golfball"
./main FCPS/Golfball.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Golfball.csv 10000 10 256 8
python3 result_parser.py

echo "Hepta"
./main FCPS/Hepta.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Hepta.csv 10000 10 256 8
python3 result_parser.py

echo "Lsun"
./main FCPS/Lsun.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Lsun.csv 10000 10 256 8
python3 result_parser.py

echo "Target"
./main FCPS/Target.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Target.csv 10000 10 256 8
python3 result_parser.py

echo "Tetra"
./main FCPS/Tetra.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/Tetra.csv 10000 10 256 8
python3 result_parser.py

echo "TwoDiamonds"
./main FCPS/TwoDiamonds.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/TwoDiamonds.csv 10000 10 256 8
python3 result_parser.py

echo "WingNut"
./main FCPS/WingNut.csv 10000 10 16 8
python3 result_parser.py
./main FCPS/WingNut.csv 10000 10 256 8
python3 result_parser.py

