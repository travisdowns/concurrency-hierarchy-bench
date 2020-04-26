include config.mk

# rebuild when makefile changes
-include dummy.rebuild

EXE := bench

.PHONY: all clean

CXX ?= g++
CC ?= gcc
ASM ?= nasm
ASM_FLAGS ?= -DNASM_ENABLE_DEBUG=$(NASM_DEBUG) -w+all

ARCH_FLAGS := -march=$(CPU_ARCH)


# make submakes use the specified compiler also
export CXX
export CC

INCLUDES += -Ifmt/include

COMMON_FLAGS := -MMD -Wall $(ARCH_FLAGS) -g $(O_LEVEL) $(INCLUDES) $(NDEBUG)

CPPFLAGS +=
CFLAGS += $(COMMON_FLAGS)
CXXFLAGS += $(COMMON_FLAGS) -Wno-unused-variable 

SRC_FILES := $(wildcard *.cpp) $(wildcard *.c) fmt/src/format.cc

# on most compilers we should use no-pie since the nasm stuff isn't position independent
# but since old compilers don't support it, you can override it with PIE= on the command line
PIE ?= -no-pie
LDFLAGS += $(PIE) -lpthread

EXTRA_DEPS :=

OBJECTS := $(SRC_FILES:.cpp=.o) $(ASM_FILES:.asm=.o)
OBJECTS := $(OBJECTS:.cc=.o)
OBJECTS := $(OBJECTS:.c=.o)
DEPFILES = $(OBJECTS:.o=.d)
# $(info OBJECTS=$(OBJECTS))

# VPATH = test:$(PSNIP_DIR)/cpu

###########
# Targets #
###########

all: bench

-include $(DEPFILES)

clean:
	rm -f *.d *.o $(EXE)

$(EXE): $(OBJECTS) $(EXTRA_DEPS)
	$(CXX) $(OBJECTS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)  -o $@

util/seqtest: util/seqtest.o

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<


%.o: %.asm
	$(ASM) $(ASM_FLAGS) -f elf64 $<

LOCAL_MK = $(wildcard local.mk)

# https://stackoverflow.com/a/3892826/149138
dummy.rebuild: Makefile config.mk $(LOCAL_MK)
	touch $@
	$(MAKE) -s clean
