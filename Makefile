VERSION = 0.0.1

BUILD_DIR = build
SOURCE_DIR = src
HEADER_DIR = src

CC := $(if $(shell which clang 2>/dev/null),clang,gcc)
NVCC = nvcc
CFLAGS = -Wall -Wextra -Werror -g
# CUDA_FLAGS = -gencode arch=compute_75,code=sm_75
# The following flag allows to interpret the c files as a CUDA files (.cu)
# CUDA_FLAGS = -x cu
LDFLAGS = 
LDFLAGS_CUDA = -lcuda -lcudart 
TARGET = SVM
# SRCS = SVM.c utils.c
SRCS = $(wildcard $(SOURCE_DIR)/*.c)
# SRCS := $(SRCS:$(SOURCE_DIR)/%=%)
OBJS = $(patsubst $(SOURCE_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
CUDA_OBJS = $(patsubst $(SOURCE_DIR)/%.c,$(BUILD_DIR)/%.cu.o,$(SRCS))

ECHO = echo
MKDIR = mkdir

all: .setup_done .update

.update: $(OBJS)
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -I$(SOURCE_DIR) -o $(TARGET) $(OBJS) $(LDFLAGS)
	@rm -f .cuda_update
	@touch .update

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.c
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -I$(SOURCE_DIR) -c $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cu.o: $(SOURCE_DIR)/%.c
	$(NVCC) $(CUDA_FLAGS) -x cu -I$(HEADER_DIR) -I$(SOURCE_DIR) -c $< -o $@ $(LDFLAGS_CUDA)

cuda: .setup_done .cuda_update

.cuda_update: $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) -I$(HEADER_DIR) -I$(SOURCE_DIR) -o $(TARGET) $(CUDA_OBJS) $(LDFLAGS_CUDA)
	@rm -f .update
	@touch .cuda_update

debug: CFLAGS += -DDEBUG
debug: .silent_clean .setup_done .update
debug: 
	@rm -f .cuda_update .update

.silent_clean:
	@rm -f $(OBJS) $(CUDA_OBJS) $(TARGET) .cuda_update .update .setup_done
	@rm -rf $(BUILD_DIR)

clean:
	@$(ECHO) "Cleaning up..."
	@rm -f $(OBJS) $(CUDA_OBJS) $(TARGET) .cuda_update .update .setup_done
	@rm -rf $(BUILD_DIR)

.setup_done: 
	@$(ECHO) "Setting up compile environment for CUDA-SVM v$(VERSION)..."
	@$(MKDIR) -p $(BUILD_DIR)
	@touch .setup_done

.PHONY: all cuda debug clean help

help:
	@$(ECHO) "Usage: make [all|CUDA|debug|clean|help]"
	@$(ECHO) "  all:    build the project for CPU (default)"
	@$(ECHO) "  cuda:   build the project with CUDA"
	@$(ECHO) "  debug:  build the project with debug flag"
	@$(ECHO) "  clean:  remove object files and executable"
	@$(ECHO) "  help:   show this message"
