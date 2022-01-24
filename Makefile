CC = 
MPICC = 
CFLAGS = 
LDFLAGS = 
LIBS = -lm
CFILES = solve_cpu.c
MPIFILES = main.c solve.c
HFILES = common.h solve.h solve_cpu.h
OBJECTS = $(CFILES:.c=.o) $(MPIFILES:.c=.o)
TARGET = prog
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(MPICC) $(LDFLAGS) $^ -o $@ $(LIBS) 
main.o: main.c
	$(MPICC) $(CFLAGS) -c -o $@ $^
solve.o: solve.c
	$(MPICC) $(CFLAGS) -c -o $@ $^
clean:
	rm -f $(TARGET) *.o prog_*
