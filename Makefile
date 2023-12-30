.PHONY: all
all:
	g++ -ggdb -O2 main.cpp -fsanitize=address
