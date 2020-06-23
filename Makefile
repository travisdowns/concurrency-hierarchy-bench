include config.mk

# rebuild when makefile changes
-include dummy.rebuild

EXE := bench

.PHONY: all clean

CXX ?= g++
CC ?= gcc

# make submakes use the specified compiler also
export CXX
export CC

# any file that is only conditionally compiled goes here,
# we filter it out from the wildcard below and then add
# it back in using COND_SRC, which gets built up based
# on various conditions
CONDSRC_MASTER := tsc-support.cpp cpuid.cpp
CONDSRC :=

ifneq ($(USE_RDTSC),0)
CONDSRC += tsc-support.cpp cpuid.cpp
endif

DEFINES = -DUSE_RDTSC=$(USE_RDTSC)

INCLUDES += -Ifmt/include

ARCH_FLAGS := $(MARCH_ARG)=$(CPU_ARCH)

COMMON_FLAGS := -MMD -Wall $(ARCH_FLAGS) -g $(O_LEVEL) $(INCLUDES) $(NDEBUG)

CFLAGS   += $(DEFINES) $(COMMON_FLAGS)
CXXFLAGS += $(DEFINES) $(COMMON_FLAGS) -Wno-unused-variable

SRC_FILES := $(wildcard *.cpp) $(wildcard *.c) fmt/src/format.cc
SRC_FILES := $(filter-out $(CONDSRC_MASTER), $(SRC_FILES)) $(CONDSRC)

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

###########
# Targets #
###########

all: bench

-include $(DEPFILES)

clean:
	find -name '*.o' -delete
	find -name '*.d' -delete
	rm -f $(EXE)

$(EXE): $(OBJECTS) $(EXTRA_DEPS)
	$(CXX) $(OBJECTS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)  -o $@

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<


%.o: %.asm
	$(ASM) $(ASM_FLAGS) -f elf64 $<

LOCAL_MK = $(wildcard local.mk)

ifndef MAKE_CLEAN_RECURSION
# https://stackoverflow.com/a/3892826/149138
dummy.rebuild: Makefile config.mk $(LOCAL_MK)
	touch $@
	$(MAKE) -s clean MAKE_CLEAN_RECURSION=1
endif
