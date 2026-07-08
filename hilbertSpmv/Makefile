# Makefile — Hilbert Curve Reordering for GPU SpMV


CC=g++
NVCC  = nvcc
LINK  = nvcc
#NVCC = /usr/local/cuda-13.0/bin/nvcc
#LINK = /usr/local/cuda-13.0/bin/nvcc
 
# ─── Architecture flags ───────────────────────────────────────
#CUDA_ARCH_FLAGS = -gencode arch=compute_70,code=sm_70   # V100   (Volta)
#CUDA_ARCH_FLAGS = -gencode arch=compute_75,code=sm_75   # RTX 2080 Super (Turing)
CUDA_ARCH_FLAGS  = -gencode arch=compute_86,code=sm_86   # RTX 3090 (Ampere)
#CUDA_ARCH_FLAGS = -gencode arch=compute_90,code=sm_90   # H100 (Hopper)
 
# ─── Compiler flags ───────────────────────────────────────────
NVCCFLAGS = --use_fast_math $(CUDA_ARCH_FLAGS) -std=c++17 -O3
 
#NVCCFLAGS = -g -G $(CUDA_ARCH_FLAGS) -std=c++17          # debug mode
 
INCPATH   = -I. -I/usr/local/cuda-13.0/include
LIBPATH   = -L/usr/local/cuda-13.0/lib64
LIBS      = -lcusparse
 
SRC_DIR   = src
BUILD_DIR   = build

REORDER_OBJ = $(BUILD_DIR)/reorder.o

# ─── Targets ──────────────────────────────────────────────────
TARGET_NAIVE          = $(BUILD_DIR)/naiveCsr
TARGET_WARP           = $(BUILD_DIR)/warpCsr
TARGET_SHARED         = $(BUILD_DIR)/sharedCsr
TARGET_BASELINE       = $(BUILD_DIR)/baseline
TARGET_ROW_REORDER    = $(BUILD_DIR)/rowReorder
TARGET_BLOCKED_HILBERT= $(BUILD_DIR)/blockedHilbert
TARGET_GLOBAL_HILBERT = $(BUILD_DIR)/globalHilbert
 
ALL_TARGETS = \
    $(TARGET_NAIVE) \
    $(TARGET_WARP) \
    $(TARGET_SHARED) \
    $(TARGET_BASELINE) \
    $(TARGET_ROW_REORDER) \
    $(TARGET_BLOCKED_HILBERT) \
    $(TARGET_GLOBAL_HILBERT)
 
# ─── Default ──────────────────────────────────────────────────
.PHONY: all clean
 
all: $(BUILD_DIR) $(REORDER_OBJ) $(ALL_TARGETS)
 
$(BIN_DIR):
	mkdir -p $(BUILD_DIR)

# ─── Custom kernels (no cuSPARSE) ─────────────────────────────
$(TARGET_NAIVE): $(SRC_DIR)/naiveCsr.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< -o $@
 
$(TARGET_WARP): $(SRC_DIR)/warpCsr.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< -o $@
 
$(TARGET_SHARED): $(SRC_DIR)/sharedCsr.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< -o $@
 
# ─── cuSPARSE baseline ────────────────────────────────────────
$(TARGET_BASELINE): $(SRC_DIR)/baseline.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $< -o $@
 
# ─── Reordering strategies ────────────────────────────────────
$(REORDER_OBJ): $(SRC_DIR)/reorder.cu $(SRC_DIR)/reorder.h Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) -c $< -o $@

$(TARGET_ROW_REORDER): $(SRC_DIR)/rowReorder.cu $(REORDER_OBJ) Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< $(REORDER_OBJ) -o $@

$(TARGET_BLOCKED_HILBERT): $(SRC_DIR)/blockedHilbert.cu $(REORDER_OBJ) Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< $(REORDER_OBJ) -o $@

$(TARGET_GLOBAL_HILBERT): $(SRC_DIR)/globalHilbert.cu $(REORDER_OBJ) Makefile
	$(NVCC) $(NVCCFLAGS) $(INCPATH) $(LIBPATH) $< $(REORDER_OBJ) -o $@
 
# ─── Clean ────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)