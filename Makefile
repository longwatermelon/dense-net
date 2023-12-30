.PHONY: all
all:
	g++ -std=c++17 -O2 -ggdb main.cpp -fsanitize=address
