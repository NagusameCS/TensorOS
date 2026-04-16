#!/usr/bin/env python3
"""Wire ott_hs_cache_flush() into axiom_beta_run and add hit stats logging."""
import os, sys

path = r"C:\Users\legom\HyperTensor\runtime\nn\axiom_beta.c"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

orig_size = len(src)

# 1. Insert flush call after memset(report, 0, sizeof(*report));
anchor1 = "    memset(report, 0, sizeof(*report));\n    report->beta_version = 3;"
replacement1 = ("    memset(report, 0, sizeof(*report));\n"
                "    ott_hs_cache_flush(); /* reset hidden-state cache each run */\n"
                "    report->beta_version = 3;")

if anchor1 not in src:
    print("ERROR: anchor1 not found — flush insertion failed")
    sys.exit(1)

src = src.replace(anchor1, replacement1, 1)
print("  + Inserted ott_hs_cache_flush() call")

# 2. Add stats log before the final return AXIOM_BETA_OK of axiom_beta_run
# The final tensor_free line just before the last return in axiom_beta_run
anchor2 = ("    tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);\n"
           "    *out_token = best_tok;\n"
           "    return AXIOM_BETA_OK;\n"
           "}\n"
           "const char *axiom_beta_status_string(")

replacement2 = ("    tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);\n"
                "    *out_token = best_tok;\n"
                "    kprintf(\"[OTT-HS] cache hits=%d misses=%d (%.1f%% hit rate)\\n\",\n"
                "            ott_hs_hits, ott_hs_misses,\n"
                "            ott_hs_hits + ott_hs_misses > 0\n"
                "                ? 100.0 * ott_hs_hits / (ott_hs_hits + ott_hs_misses)\n"
                "                : 0.0);\n"
                "    return AXIOM_BETA_OK;\n"
                "}\n"
                "const char *axiom_beta_status_string(")

if anchor2 not in src:
    print("ERROR: anchor2 not found — stats log insertion failed")
    sys.exit(1)

src = src.replace(anchor2, replacement2, 1)
print("  + Added cache hit stats log before final return")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"Done. Size delta: {len(src) - orig_size:+d} bytes")
