.PHONY: all
all:
	g++ -std=c++17 -ggdb -O2 main.cpp -fsanitize=address
