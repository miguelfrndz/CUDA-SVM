CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -Werror -g
# CUDA_FLAGS = -gencode arch=compute_75,code=sm_75
CUDA_FLAGS = -x cu
LDFLAGS = 
LDFLAGS_CUDA = -lcuda -lcudart 
TARGET = SVM
SRCS = SVM.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

CUDA: $(SRCS)
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS_CUDA)

debug: CFLAGS += -DDEBUG
debug: clean
debug: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
