# csrc/slbt/Makefile

CC = gcc
CFLAGS = -shared -fPIC -O3 -Wall -Wextra
SOURCES = slbt_core.c slbt_als.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = libslbt.dylib
INSTALL_DIR = ../../slbt/_backend

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    TARGET = libslbt.so
endif
ifeq ($(UNAME_S),Darwin)
    TARGET = libslbt.dylib
endif
ifeq ($(UNAME_S),Windows_NT)
    TARGET = libslbt.dll
    CC = gcc
endif

# Targets
.PHONY: all install clean test info debug

all: $(TARGET)

$(TARGET): $(SOURCES) slbt.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)
	@echo "✓ Built $(TARGET)"

install: $(TARGET)
	@mkdir -p $(INSTALL_DIR)
	@cp $(TARGET) $(INSTALL_DIR)/
	@echo "✓ Installed $(TARGET) to $(INSTALL_DIR)"

clean:
	@rm -f $(TARGET) $(OBJECTS)
	@rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "✓ Cleaned build files"

test: install
	@echo "Testing SLBT library..."
	@cd ../.. && python -c "from slbt import SLBT; print('✓ SLBT imported successfully')"

debug: CFLAGS += -g -DDEBUG
debug: clean all
	@echo "✓ Built debug version"

info:
	@echo "Configuration:"
	@echo "  OS:         $(UNAME_S)"
	@echo "  CC:         $(CC)"
	@echo "  CFLAGS:     $(CFLAGS)"
	@echo "  TARGET:     $(TARGET)"
	@echo "  SOURCES:    $(SOURCES)"
	@echo "  INSTALL:    $(INSTALL_DIR)"