/* =============================================================================
 * TensorOS - AI Shell Implementation
 * =============================================================================*/

#include "userland/shell/aishell.h"
#include "kernel/core/perf.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/update/ota.h"
#include "runtime/nn/llm.h"

/* ---- Forward declarations ---- */
static int  shell_exec_builtin(aishell_t *sh, int argc, char **argv);
static void shell_parse_line(const char *line, int *argc, char *argv[]);
static void shell_print_banner(void);
static void shell_print_help(void);

/* ---- Helpers ---- */

static int shell_strcmp(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *(unsigned char *)a - *(unsigned char *)b;
}

__attribute__((unused))
static int shell_strncmp(const char *a, const char *b, uint64_t n)
{
    while (n && *a && *a == *b) { a++; b++; n--; }
    return n == 0 ? 0 : *(unsigned char *)a - *(unsigned char *)b;
}

static uint64_t shell_strlen(const char *s)
{
    uint64_t n = 0;
    while (s[n]) n++;
    return n;
}

static void shell_strcpy(char *dst, const char *src)
{
    while (*src) *dst++ = *src++;
    *dst = 0;
}

/* ---- Console I/O ---- */

/* Use the interrupt-driven keyboard driver from klib */
static char read_key(void)
{
    return keyboard_getchar();
}

/* ---- History ---- */

static void history_add(shell_history_t *h, const char *line)
{
    if (shell_strlen(line) == 0) return;
    int idx = h->count % SHELL_MAX_HISTORY;
    shell_strcpy(h->lines[idx], line);
    h->count++;
    h->cursor = h->count;
}

/* ---- Line reading ---- */

static int shell_read_line(aishell_t *sh, char *buf, int max)
{
    int pos = 0;
    while (pos < max - 1) {
        char c = read_key();
        if (c == 0) continue;
        if (c == '\n' || c == '\r') {
            buf[pos] = '\0';
            kprintf("\n");
            return pos;
        }
        if (c == '\b' || c == 0x7F) {
            if (pos > 0) {
                pos--;
                kprintf("\b \b");
            }
            continue;
        }
        buf[pos++] = c;
        /* Echo */
        char echo[2] = {c, 0};
        kprintf("%s", echo);
    }
    buf[pos] = '\0';
    return pos;
}

/* ---- Parser: split line into argv ---- */

static void shell_parse_line(const char *line, int *argc, char *argv[])
{
    *argc = 0;
    const char *p = line;
    static char token_buf[SHELL_MAX_LINE];
    char *t = token_buf;

    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0') break;

        argv[*argc] = t;

        if (*p == '"') {
            /* Quoted string */
            p++;
            while (*p && *p != '"') *t++ = *p++;
            if (*p == '"') p++;
        } else {
            while (*p && *p != ' ' && *p != '\t') *t++ = *p++;
        }
        *t++ = '\0';
        (*argc)++;
        if (*argc >= SHELL_MAX_ARGS) break;
    }
}

/* ---- Built-in commands ---- */

static int cmd_model(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: model <load|list|info|kill> [args...]\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "list") == 0) {
        kprintf("ID   STATE      NAME                 FLOPS        MEM\n");
        kprintf("---  ---------  -------------------  -----------  -----------\n");
        for (uint32_t i = 0; i < kstate.meu_count; i++) {
            model_exec_unit_t *meu = &kstate.meus[i];
            kprintf("%3u  %-9s  %-19s  %11lu  %11lu\n",
                    (uint32_t)meu->meu_id,
                    meu->state == MEU_STATE_RUNNING ? "RUNNING" :
                    meu->state == MEU_STATE_READY   ? "READY"   :
                    meu->state == MEU_STATE_LOADING ? "LOADING" : "OTHER",
                    meu->name,
                    meu->flops,
                    meu->vram_used);
        }
        kprintf("\nTotal MEUs: %d\n", kstate.meu_count);
        return 0;
    }

    if (shell_strcmp(argv[1], "load") == 0) {
        if (argc < 3) {
            kprintf("Usage: model load <name>\n");
            return 1;
        }
        kprintf("[SHELL] Loading model '%s'...\n", argv[2]);
        /* TODO: Interface with scheduler to create MEU */
        kprintf("[SHELL] Model loaded as MEU #%d\n", kstate.meu_count);
        return 0;
    }

    if (shell_strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            kprintf("Usage: model info <id>\n");
            return 1;
        }
        kprintf("[SHELL] MEU info placeholder\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "kill") == 0) {
        if (argc < 3) {
            kprintf("Usage: model kill <id>\n");
            return 1;
        }
        kprintf("[SHELL] Killing MEU placeholder\n");
        return 0;
    }

    kprintf("Unknown model subcommand: %s\n", argv[1]);
    return 1;
}

static int cmd_tensor(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: tensor <shape|cast|info> [args...]\n");
        return 1;
    }
    kprintf("[SHELL] Tensor command: %s\n", argv[1]);
    return 0;
}

static int cmd_infer(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) {
        kprintf("Usage: infer <model> <input>\n");
        return 1;
    }
    kprintf("[SHELL] Running inference: model=%s input=%s\n", argv[1], argv[2]);
    kprintf("[SHELL] Dispatching to tensor engine...\n");
    /* TODO: Create MEU, load model, run forward pass */
    return 0;
}

static int cmd_train(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) {
        kprintf("Usage: train <model> <dataset>\n");
        return 1;
    }
    kprintf("[SHELL] Launching training: model=%s dataset=%s\n", argv[1], argv[2]);
    return 0;
}

static int cmd_deploy(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: deploy <model> [port]\n");
        return 1;
    }
    int port = argc >= 3 ? 8080 : 8080; /* TODO: parse port */
    kprintf("[SHELL] Deploying model '%s' on port %d\n", argv[1], port);
    return 0;
}

static int cmd_git(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: git <init|commit|log|status|diff|branch|push|pull|clone>\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "init") == 0) {
        git_repo_t repo;
        int r = git_repo_init("/", &repo);
        kprintf(r == 0 ? "Initialized git repository\n" : "Failed to init repo\n");
        return r;
    }

    if (shell_strcmp(argv[1], "commit") == 0) {
        const char *msg = argc >= 4 && shell_strcmp(argv[2], "-m") == 0 ? argv[3] : "no message";
        kprintf("[GIT] Creating commit: %s\n", msg);
        return 0;
    }

    if (shell_strcmp(argv[1], "log") == 0) {
        kprintf("[GIT] Commit log placeholder\n");
        return 0;
    }

    if (shell_strcmp(argv[1], "status") == 0) {
        kprintf("[GIT] Status placeholder\n");
        return 0;
    }

    kprintf("Unknown git subcommand: %s\n", argv[1]);
    return 1;
}

static int cmd_pkg(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: pkg <install|search|list|remove|info> [args...]\n");
        return 1;
    }

    if (shell_strcmp(argv[1], "install") == 0 && argc >= 3) {
        kprintf("[PKG] Installing model '%s'...\n", argv[2]);
        modelpkg_install(argv[2], "/models");
        return 0;
    }

    if (shell_strcmp(argv[1], "search") == 0 && argc >= 3) {
        model_manifest_t results[16];
        uint32_t count = 0;
        modelpkg_search(argv[2], results, 16, &count);
        kprintf("Found %d packages matching '%s'\n", count, argv[2]);
        return 0;
    }

    if (shell_strcmp(argv[1], "list") == 0) {
        kprintf("[PKG] Installed models placeholder\n");
        return 0;
    }

    kprintf("Unknown pkg subcommand: %s\n", argv[1]);
    return 1;
}

static int cmd_sandbox(aishell_t *sh, int argc, char **argv)
{
    if (argc < 3) {
        kprintf("Usage: sandbox <strict|standard|permissive> <command...>\n");
        return 1;
    }
    kprintf("[SHELL] Running in sandbox (policy=%s): %s\n", argv[1], argv[2]);
    return 0;
}

static int cmd_monitor(aishell_t *sh, int argc, char **argv)
{
    kprintf("=== TensorOS System Monitor ===\n");
    kprintf("Uptime:          %lu ticks\n", kstate.uptime_ticks);
    kprintf("MEUs running:    %d\n", kstate.meu_count);
    kprintf("GPUs detected:   %d\n", kstate.gpu_count);
    kprintf("Tensor ops:      %lu\n", kstate.tensor_ops_total);
    kprintf("Memory used:     %lu MB\n", kstate.memory_used_bytes / (1024*1024));
    kprintf("Memory total:    %lu MB\n", kstate.memory_total_bytes / (1024*1024));
    return 0;
}

static int cmd_run(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: run <script.pseudo>\n");
        return 1;
    }
    kprintf("[SHELL] Executing Pseudocode script: %s\n", argv[1]);
    /* TODO: Read file from TensorFS, feed to pseudocode JIT */
    return 0;
}

/* ---- New functional commands ---- */

static int cmd_bench(aishell_t *sh, int argc, char **argv)
{
    extern void run_benchmarks(void);
    kprintf("[BENCH] Running performance benchmarks...\n\n");
    run_benchmarks();
    return 0;
}

static int cmd_demo(aishell_t *sh, int argc, char **argv)
{
    if (argc < 2) {
        kprintf("Usage: demo <type>\n");
        kprintf("  infer    Neural network inference demos\n");
        kprintf("  quant    INT16 quantized inference\n");
        kprintf("  evolve   Neuroevolution architecture search\n");
        kprintf("  train    Backprop + Adam training\n");
        kprintf("  sne      Speculative Neural Execution (5 techniques)\n");
        kprintf("  transformer  KV-cache transformer engine\n");
        kprintf("  q4       INT4 block quantization\n");
        kprintf("  arena    Tensor memory arena\n");
        kprintf("  mathllm  Micro math LLMs\n");
        kprintf("  gguf     GGUF model parser demo\n");
        kprintf("  llmeval  Full LLM benchmark + math eval\n");
        kprintf("  all      Run everything\n");
        return 1;
    }
    extern void nn_run_demos(void);
    extern void nn_quant_demos(void);
    extern void nn_evolve_demos(void);
    extern void nn_train_demos(void);
    extern void sne_run_demos(void);
    extern void tf_run_demos(void);
    extern void q4_run_demos(void);
    extern void arena_run_demos(void);
    extern void math_llm_run_eval(void);
    extern void gguf_run_demos(void);
    extern void llm_run_full_eval(void);

    if (shell_strcmp(argv[1], "infer") == 0)       { nn_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "quant") == 0)        { nn_quant_demos(); return 0; }
    if (shell_strcmp(argv[1], "evolve") == 0)       { nn_evolve_demos(); return 0; }
    if (shell_strcmp(argv[1], "train") == 0)        { nn_train_demos(); return 0; }
    if (shell_strcmp(argv[1], "sne") == 0)          { sne_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "transformer") == 0)  { tf_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "q4") == 0)           { q4_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "arena") == 0)        { arena_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "mathllm") == 0)      { math_llm_run_eval(); return 0; }
    if (shell_strcmp(argv[1], "gguf") == 0)         { gguf_run_demos(); return 0; }
    if (shell_strcmp(argv[1], "llmeval") == 0)      { llm_run_full_eval(); return 0; }
    if (shell_strcmp(argv[1], "all") == 0) {
        nn_run_demos();
        nn_quant_demos();
        nn_evolve_demos();
        nn_train_demos();
        sne_run_demos();
        tf_run_demos();
        q4_run_demos();
        arena_run_demos();
        math_llm_run_eval();
        gguf_run_demos();
        return 0;
    }
    kprintf("Unknown demo: %s (type 'demo' for list)\n", argv[1]);
    return 1;
}

static int cmd_status(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t perf_tsc_mhz(void);
    extern uint64_t tensor_mm_free_bytes(void);

    kprintf("\n=== TensorOS Status ===\n");
    kprintf("  Version:    v0.1.0 \"Neuron\"\n");
    kprintf("  Phase:      %s\n",
            kstate.phase == 0 ? "BOOT" :
            kstate.phase == 1 ? "INIT" :
            kstate.phase == 2 ? "RUNNING" : "PANIC");
    kprintf("  CPUs:       %d\n", kstate.cpu_count);
    kprintf("  GPUs:       %d\n", kstate.gpu_count);
    kprintf("  TPUs:       %d\n", kstate.tpu_count);
    kprintf("  TSC:        %lu MHz\n", perf_tsc_mhz());
    kprintf("  Memory:     %lu MB free\n", tensor_mm_free_bytes() / (1024*1024));
    kprintf("  MEUs:       %d running\n", kstate.meu_count);
    kprintf("  Tensor ops: %lu total\n", kstate.tensor_ops_total);
    kprintf("  SIMD:       SSE2 (4-wide float)\n");
    kprintf("  Shell cmds: %u executed\n", sh->commands_executed);
    kprintf("\n");
    return 0;
}

static int cmd_ota(aishell_t *sh, int argc, char **argv)
{
    kprintf("[OTA] Entering chain-load receive mode (RAM only, no SD write)\n");
    kprintf("[OTA] Use push_rpi.ps1 on your PC to send a new kernel\n");
    int r = ota_receive_and_chainload();
    /* Only returns on error */
    kprintf("[OTA] Failed (error %d)\n", r);
    return r;
}

static int cmd_flash(aishell_t *sh, int argc, char **argv)
{
    kprintf("[FLASH] Entering persistent flash mode (writes to SD card)\n");
    kprintf("[FLASH] Use push_rpi.ps1 -Flash on your PC to send a new kernel\n");
    int r = ota_receive_and_flash();
    /* Only returns on error */
    kprintf("[FLASH] Failed (error %d)\n", r);
    return r;
}

static int cmd_clear(aishell_t *sh, int argc, char **argv)
{
    vga_init();
    return 0;
}

static int cmd_uname(aishell_t *sh, int argc, char **argv)
{
#if defined(__aarch64__)
    kprintf("TensorOS v0.1.0 \"Neuron\" aarch64 NEON\n");
#else
    kprintf("TensorOS v0.1.0 \"Neuron\" x86_64 SSE2\n");
#endif
    return 0;
}

/* ---- Interactive LLM prompt ---- */

static int cmd_llm(aishell_t *sh, int argc, char **argv)
{
    if (!llm_is_loaded()) {
        kprintf("[LLM] No model loaded. Attach a GGUF model disk to use this.\n");
        return 1;
    }

    if (argc < 2) {
        kprintf("[LLM] Model: %s\n", llm_model_name());
        kprintf("Usage: llm <your prompt text>\n");
        kprintf("  e.g. llm What is 2+2?\n");
        kprintf("  e.g. llm Explain neural networks briefly\n");
        return 0;
    }

    /* Concatenate all argv into one prompt string */
    static char prompt[1024];
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (i > 1 && pos < 1020) prompt[pos++] = ' ';
        for (const char *p = argv[i]; *p && pos < 1020; p++)
            prompt[pos++] = *p;
    }
    prompt[pos] = '\0';

    kprintf("[LLM] Model: %s\n", llm_model_name());
    kprintf("[LLM] Prompt: %s\n", prompt);
    kprintf("[LLM] Generating...\n");

    static char response[2048];
    uint64_t t0 = rdtsc_fenced();
    int n_gen = llm_prompt(prompt, response, sizeof(response));
    uint64_t t1 = rdtsc_fenced();
    uint64_t ms = perf_cycles_to_us(t1 - t0) / 1000;

    if (n_gen < 0) {
        kprintf("[LLM] Error generating response.\n");
        return 1;
    }

    kprintf("\n%s\n\n", response);
    kprintf("[LLM] %d tokens in %lu ms", n_gen, ms);
    if (n_gen > 0 && ms > 0) {
        kprintf(" (%lu ms/tok)", ms / (uint64_t)n_gen);
    }
    kprintf("\n");
    return 0;
}

/* ---- Reset LLM conversation ---- */

static int cmd_reset(aishell_t *sh, int argc, char **argv)
{
    extern void llm_reset_cache(void);
    llm_reset_cache();
    kprintf("[LLM] Conversation reset. KV cache cleared.\n");
    return 0;
}

/* ---- Memory info ---- */

static int cmd_mem(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t tensor_mm_free_bytes(void);
    kprintf("\n=== Memory ===\n");
    kprintf("  Total:     %lu MB\n", kstate.memory_total_bytes / (1024*1024));
    kprintf("  Used:      %lu MB\n", kstate.memory_used_bytes / (1024*1024));
    kprintf("  Free:      %lu MB\n", tensor_mm_free_bytes() / (1024*1024));
    kprintf("  Tensor heap: 256 MB\n");
    kprintf("  Model cache: 512 MB\n");
    kprintf("\n");
    return 0;
}

/* ---- Uptime ---- */

static int cmd_uptime(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t watchdog_uptime_ms(void);
    uint64_t ms = watchdog_uptime_ms();
    uint64_t sec = ms / 1000;
    uint64_t min = sec / 60;
    uint64_t hr  = min / 60;
    kprintf("up %lu:%02lu:%02lu (%lu ms)\n", hr, min % 60, sec % 60, ms);
    return 0;
}

/* ---- Reboot ---- */

static int cmd_reboot(aishell_t *sh, int argc, char **argv)
{
    kprintf("[REBOOT] Resetting system...\n");
#ifndef __aarch64__
    /* Triple-fault: load invalid IDT and trigger interrupt */
    __asm__ volatile ("cli");
    struct { uint16_t limit; uint64_t base; } __attribute__((packed)) null_idtr = {0, 0};
    __asm__ volatile ("lidt %0" : : "m"(null_idtr));
    __asm__ volatile ("int $0x03");
#else
    /* ARM64: use PSCI SYSTEM_RESET */
    __asm__ volatile ("mov x0, #0x84000009; hvc #0");
#endif
    /* Should not reach here */
    while(1) __asm__ volatile ("hlt");
    return 0;
}

/* ---- Process / MEU list ---- */

static int cmd_ps(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n  PID  STATE      TYPE       NAME\n");
    kprintf("  ---  ---------  ---------  --------------------\n");
    kprintf("  0    RUNNING    kernel     TensorOS kernel\n");
    kprintf("  1    RUNNING    shell      aishell\n");
    for (uint32_t i = 0; i < kstate.meu_count; i++) {
        model_exec_unit_t *meu = &kstate.meus[i];
        kprintf("  %3u  %-9s  model      %s\n",
                i + 2,
                meu->state == MEU_STATE_RUNNING ? "RUNNING" :
                meu->state == MEU_STATE_READY   ? "READY"   :
                meu->state == MEU_STATE_LOADING ? "LOADING" : "STOPPED",
                meu->name);
    }
    if (llm_is_loaded()) {
        kprintf("  %3u  RUNNING    llm        %s\n", kstate.meu_count + 2, llm_model_name());
    }
    kprintf("\n");
    return 0;
}

/* ---- Echo ---- */

static int cmd_echo(aishell_t *sh, int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (i > 1) kprintf(" ");
        kprintf("%s", argv[i]);
    }
    kprintf("\n");
    return 0;
}

/* ---- lspci ---- */

static int cmd_lspci(aishell_t *sh, int argc, char **argv)
{
    kprintf("\n=== PCI Devices ===\n");
    extern void pci_enumerate(void);
    pci_enumerate();
    kprintf("\n");
    return 0;
}

/* ---- sysinfo (comprehensive) ---- */

static int cmd_sysinfo(aishell_t *sh, int argc, char **argv)
{
    extern uint64_t perf_tsc_mhz(void);
    extern uint64_t tensor_mm_free_bytes(void);
    extern uint64_t watchdog_uptime_ms(void);

    uint64_t ms  = watchdog_uptime_ms();
    uint64_t sec = ms / 1000;

    kprintf("\n=== TensorOS System Information ===\n\n");
#if defined(__aarch64__)
    kprintf("  OS:         TensorOS v0.1.0 \"Neuron\" (aarch64)\n");
    kprintf("  SIMD:       NEON 128-bit\n");
#else
    kprintf("  OS:         TensorOS v0.1.0 \"Neuron\" (x86_64)\n");
    kprintf("  SIMD:       SSE2 128-bit\n");
#endif
    kprintf("  CPUs:       %d\n", kstate.cpu_count);
    kprintf("  GPUs:       %d\n", kstate.gpu_count);
    kprintf("  TSC:        %lu MHz\n", perf_tsc_mhz());
    kprintf("  RAM:        %lu MB total, %lu MB free\n",
            kstate.memory_total_bytes / (1024*1024),
            tensor_mm_free_bytes() / (1024*1024));
    kprintf("  Uptime:     %lu s\n", sec);
    kprintf("  MEUs:       %d loaded\n", kstate.meu_count);
    kprintf("  Tensor ops: %lu\n", kstate.tensor_ops_total);
    if (llm_is_loaded())
        kprintf("  LLM:        %s (loaded)\n", llm_model_name());
    else
        kprintf("  LLM:        (none)\n");
    kprintf("\n");
    return 0;
}

static void shell_print_help(void)
{
    kprintf("\n");
    kprintf("TensorOS AI Shell - Built-in Commands\n");
    kprintf("======================================\n");
    kprintf("  --- General ---\n");
    kprintf("  help                      This message\n");
    kprintf("  status                    System status overview\n");
    kprintf("  sysinfo                   Comprehensive system info\n");
    kprintf("  mem                       Memory usage\n");
    kprintf("  uptime                    System uptime\n");
    kprintf("  ps                        List processes / MEUs\n");
    kprintf("  lspci                     List PCI devices\n");
    kprintf("  uname                     Show OS version\n");
    kprintf("  echo <text>               Print text\n");
    kprintf("  clear                     Clear screen\n");
    kprintf("  reboot                    Restart system\n");
    kprintf("  exit                      Shutdown\n");
    kprintf("\n  --- AI / LLM ---\n");
    kprintf("  llm <prompt>              Chat with the loaded LLM\n");
    kprintf("  reset                     Reset LLM conversation context\n");
    kprintf("  infer <model> <input>     Run inference\n");
    kprintf("  train <model> <dataset>   Launch training\n");
    kprintf("  deploy <model> [port]     Deploy as service\n");
    kprintf("  model list|load|info|kill  Manage model exec units\n");
    kprintf("  tensor shape|cast|info    Tensor operations\n");
    kprintf("\n  --- Benchmarks & Demos ---\n");
    kprintf("  bench                     Run GEMM/SIMD benchmarks\n");
    kprintf("  demo <type>               AI demos (type 'demo' for list)\n");
    kprintf("\n  --- Subsystems ---\n");
    kprintf("  monitor                   System monitor\n");
    kprintf("  git <subcommand>          Kernel-level git\n");
    kprintf("  pkg install|search|list   Model package manager\n");
    kprintf("  sandbox <policy> <cmd>    Run in sandbox\n");
    kprintf("  run <script.pseudo>       Execute Pseudocode\n");
    kprintf("  ota                       OTA update (RAM chain-load)\n");
    kprintf("  flash                     OTA update (persistent SD write)\n");
    kprintf("\n");
    kprintf("Any other input is treated as Pseudocode and JIT-compiled.\n\n");
}

static int shell_exec_builtin(aishell_t *sh, int argc, char **argv)
{
    if (argc == 0) return 0;

    if (shell_strcmp(argv[0], "help")    == 0) { shell_print_help(); return 0; }
    if (shell_strcmp(argv[0], "status")  == 0) return cmd_status(sh, argc, argv);
    if (shell_strcmp(argv[0], "sysinfo") == 0) return cmd_sysinfo(sh, argc, argv);
    if (shell_strcmp(argv[0], "mem")     == 0) return cmd_mem(sh, argc, argv);
    if (shell_strcmp(argv[0], "uptime")  == 0) return cmd_uptime(sh, argc, argv);
    if (shell_strcmp(argv[0], "ps")      == 0) return cmd_ps(sh, argc, argv);
    if (shell_strcmp(argv[0], "lspci")   == 0) return cmd_lspci(sh, argc, argv);
    if (shell_strcmp(argv[0], "echo")    == 0) return cmd_echo(sh, argc, argv);
    if (shell_strcmp(argv[0], "reboot")  == 0) return cmd_reboot(sh, argc, argv);
    if (shell_strcmp(argv[0], "llm")     == 0) return cmd_llm(sh, argc, argv);
    if (shell_strcmp(argv[0], "bench")   == 0) return cmd_bench(sh, argc, argv);
    if (shell_strcmp(argv[0], "demo")    == 0) return cmd_demo(sh, argc, argv);
    if (shell_strcmp(argv[0], "model")   == 0) return cmd_model(sh, argc, argv);
    if (shell_strcmp(argv[0], "tensor")  == 0) return cmd_tensor(sh, argc, argv);
    if (shell_strcmp(argv[0], "infer")   == 0) return cmd_infer(sh, argc, argv);
    if (shell_strcmp(argv[0], "train")   == 0) return cmd_train(sh, argc, argv);
    if (shell_strcmp(argv[0], "deploy")  == 0) return cmd_deploy(sh, argc, argv);
    if (shell_strcmp(argv[0], "git")     == 0) return cmd_git(sh, argc, argv);
    if (shell_strcmp(argv[0], "pkg")     == 0) return cmd_pkg(sh, argc, argv);
    if (shell_strcmp(argv[0], "sandbox") == 0) return cmd_sandbox(sh, argc, argv);
    if (shell_strcmp(argv[0], "monitor") == 0) return cmd_monitor(sh, argc, argv);
    if (shell_strcmp(argv[0], "run")     == 0) return cmd_run(sh, argc, argv);
    if (shell_strcmp(argv[0], "ota")     == 0) return cmd_ota(sh, argc, argv);
    if (shell_strcmp(argv[0], "flash")   == 0) return cmd_flash(sh, argc, argv);
    if (shell_strcmp(argv[0], "clear")   == 0) return cmd_clear(sh, argc, argv);
    if (shell_strcmp(argv[0], "reset")   == 0) return cmd_reset(sh, argc, argv);
    if (shell_strcmp(argv[0], "uname")   == 0) return cmd_uname(sh, argc, argv);

    return -1; /* Not a builtin */
}

/* ---- Banner ---- */

static void shell_print_banner(void)
{
    kprintf("\n");
    kprintf("  ______                         ____  _____\n");
    kprintf(" /_  __/__  ____  _________  ___/ __ \\/ ___/\n");
    kprintf("  / / / _ \\/ __ \\/ ___/ __ \\/ _/ / / /\\__ \\ \n");
    kprintf(" / / /  __/ / / (__  ) /_/ / / / /_/ /___/ / \n");
    kprintf("/_/  \\___/_/ /_/____/\\____/_/  \\____//____/  \n");
    kprintf("\n");
    kprintf("TensorOS v0.1.0 - AI-First Operating System\n");
    kprintf("%d CPU(s), %lu MB RAM", kstate.cpu_count,
            kstate.memory_total_bytes / (1024*1024));
    if (llm_is_loaded())
        kprintf(", LLM: %s", llm_model_name());
    kprintf("\n");
    kprintf("Type 'help' for commands, 'llm <prompt>' to chat with AI.\n\n");
}

/* ---- Main Shell Entry ---- */

void aishell_init(aishell_t *sh)
{
    kmemset(sh, 0, sizeof(*sh));
    shell_strcpy(sh->prompt, "tensor> ");
    sh->running = true;
    sh->interactive = true;
    sh->session_start_ticks = kstate.uptime_ticks;

    /* Initialize embedded Pseudocode runtime for ad-hoc scripting */
    sh->runtime = (pseudo_runtime_t *)kmalloc(sizeof(pseudo_runtime_t));
    if (sh->runtime)
        kmemset(sh->runtime, 0, sizeof(*sh->runtime));
}

void aishell_run(aishell_t *sh)
{
    shell_print_banner();

    char line[SHELL_MAX_LINE];
    char *argv[SHELL_MAX_ARGS];
    int argc;

    while (sh->running) {
        kprintf("%s", sh->prompt);

        int len = shell_read_line(sh, line, SHELL_MAX_LINE);
        if (len == 0) continue;

        history_add(&sh->history, line);

        /* Check for exit */
        if (shell_strcmp(line, "exit") == 0 || shell_strcmp(line, "quit") == 0) {
            kprintf("Shutting down TensorOS...\n");
            sh->running = false;
            break;
        }

        /* Parse and try builtins */
        shell_parse_line(line, &argc, argv);
        int r = shell_exec_builtin(sh, argc, argv);

        if (r == -1) {
            /* Not a builtin — treat as Pseudocode expression */
            kprintf("[JIT] Compiling: %s\n", line);
            if (sh->runtime)
                pseudo_exec_string(sh->runtime, line);
        }

        sh->commands_executed++;
    }
}

/* Kernel entry point for shell — this runs the interactive REPL */
void aishell_main(void)
{
    static aishell_t shell;  /* Static to avoid 4KB+ on the 64KB stack */
    aishell_init(&shell);
    aishell_run(&shell);
    /* Only returns if user types 'exit' */
}
