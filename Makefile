EXEC := ./buddha

CFLAGS := -g -Og -Ofast -Wall -lm -l OpenCL 

OBJS := $(patsubst %.c,%.o,$(wildcard *.c))

build: $(OBJS)
		gcc $(OBJS) -o $(EXEC) $(CFLAGS)

# pull in dependency info for *existing* .o files
# -include $(OBJS:.o=.dep)
#
#  # compile and generate dependency info
%.o: %.c
		gcc -c $(CFLAGS) $*.c -o $*.o
		gcc -MM $(CFLAGS) $*.c > $*.dep

# remove compilation products
clean:
	rm -f $(EXEC) *.o *.dep
run: build
	$(EXEC)
