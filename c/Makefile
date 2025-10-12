CC=gcc
CFLAGS=-O3 -std=c11 -Wall -Wextra -pedantic -D_GNU_SOURCE
LDFLAGS=-ldl -lm

all: bench
bench: bench.o paragon.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f bench bench.o paragon.o
