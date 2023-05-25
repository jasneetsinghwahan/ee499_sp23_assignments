TARGETS = dump_lb 

all: $(TARGETS)
.PHONY: all

$(TARGETS): %: %.bpf.o 

%.bpf.o: %.c
	clang -I/usr/lib/modules/5.15.0-71-generic/build/include -target bpf \
		-I/usr/include/$(shell uname -m)-linux-gnu -I/usr/src/linux-headers-5.15.0-71-generic/include\
		-g -v\
	    -O2 -o $@ -c $<

clean: 
	- rm *.bpf.o
	- rm -f /sys/fs/bpf/hello 
	- rm -f /sys/fs/bpf/hello-func

