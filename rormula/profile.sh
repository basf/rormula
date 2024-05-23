maturin develop --release --features print_timings
python test/test_wilkinson.py 2> counts.txt
counts -i -e counts.txt
