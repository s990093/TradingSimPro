# Makefile for running enhanced backtest

.PHONY: all clean run

# Default target
all: run

# Run the enhanced backtest
run:
	# python src/backtest_enhanced.py --loss_threshold  0.3404658937420397  --profit_threshold 0.0

# Clean up any generated files
clean:
	rm -f res/ml_enhanced_trades.csv

