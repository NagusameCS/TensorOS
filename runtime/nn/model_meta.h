/*
 * TensorOS Model Metadata Normalization
 *
 * Maps tensor names from various model formats (HuggingFace/PyTorch,
 * safetensors) to GGUF canonical names used by the inference pipeline.
 *
 * Also provides format detection (magic number sniffing) and
 * architecture inference from tensor names/shapes.
 */

#ifndef TENSOROS_MODEL_META_H
#define TENSOROS_MODEL_META_H

#include <stdint.h>

/* ─── Model format identifiers ─── */
typedef enum {
    MODEL_FMT_UNKNOWN    = 0,
    MODEL_FMT_GGUF       = 1,
    MODEL_FMT_SAFETENSORS= 2,
    MODEL_FMT_ONNX       = 3,
    MODEL_FMT_PYTORCH    = 4,
} model_format_t;

/* ─── Architecture identifiers ─── */
typedef enum {
    MODEL_ARCH_UNKNOWN = 0,
    MODEL_ARCH_LLAMA   = 1,
    MODEL_ARCH_GEMMA   = 2,
    MODEL_ARCH_GEMMA2  = 3,
    MODEL_ARCH_GEMMA4  = 4,
    MODEL_ARCH_PHI2    = 5,
    MODEL_ARCH_PHI3    = 6,
    MODEL_ARCH_QWEN2   = 7,
    MODEL_ARCH_MISTRAL = 8,
    MODEL_ARCH_GPT2    = 9,
} model_arch_t;

/* ─── Format detection ─── */

/* Detect model format from file data (magic number sniffing).
 * Returns MODEL_FMT_UNKNOWN if format cannot be determined. */
model_format_t model_detect_format(const void *data, uint64_t size);

/* ─── Tensor name normalization ─── */

/* Normalize a HuggingFace/PyTorch tensor name to GGUF canonical form.
 * Writes the canonical name to out_name (max out_len chars).
 * Returns 0 on success, -1 if name cannot be mapped. */
int model_normalize_tensor_name(const char *hf_name, char *out_name, int out_len);

/* ─── Architecture inference ─── */

/* Infer model architecture from a set of tensor names.
 * tensor_names is an array of n_tensors strings.
 * Returns inferred architecture. */
model_arch_t model_infer_arch(const char **tensor_names, int n_tensors);

#endif /* TENSOROS_MODEL_META_H */
