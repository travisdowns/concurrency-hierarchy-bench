-include local.mk

# set DEBUG to 1 to enable various debugging checks
DEBUG ?= 0
CPP_STD ?= c++11
C_STD ?= c11
CPU_ARCH ?= native

# $(info DEBUG=$(DEBUG))

ifeq ($(DEBUG),1)
O_LEVEL ?= -O0
NASM_DEBUG ?= 1
NDEBUG=
else
O_LEVEL ?= -O2
NASM_DEBUG ?= 0
NDEBUG=-DNDEBUG
endif
