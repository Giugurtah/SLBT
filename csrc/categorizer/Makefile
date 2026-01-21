# csrc/categorizer/Makefile

CC = gcc
CFLAGS = -shared -fPIC -O3 -Wall -Wextra
SOURCES = kmeans.c kmeans_init.c kmeans_metrics.c categorizer_core.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = libcategorizer.dylib
INSTALL_DIR = ../../slbt/_preprocessing/_backend

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    TARGET = libcategorizer.so
endif
ifeq ($(UNAME_S),Darwin)
    TARGET = libcategorizer.dylib
endif
ifeq ($(UNAME_S),Windows_NT)
    TARGET = libcategorizer.dll
    CC = gcc
endif

# Targets
.PHONY: all install clean test

all: $(TARGET)

$(TARGET): $(SOURCES) kmeans.h
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

test: $(TARGET)
	@echo "Testing categorizer library..."
	@python3 test_categorizer.py

# Development targets
debug: CFLAGS += -g -DDEBUG
debug: clean all
	@echo "✓ Built debug version"

# Show configuration
info:
	@echo "Configuration:"
	@echo "  OS:         $(UNAME_S)"
	@echo "  CC:         $(CC)"
	@echo "  CFLAGS:     $(CFLAGS)"
	@echo "  TARGET:     $(TARGET)"
	@echo "  SOURCES:    $(SOURCES)"
	@echo "  INSTALL:    $(INSTALL_DIR)"