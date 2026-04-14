/*
 * tensorchat — command-line chat interface for TensorOS runtime
 *
 * Connects to a TensorOS runtime HTTP endpoint and provides an interactive
 * REPL for chat, prompt completion, and model status queries.
 *
 * Build:
 *   zig cc -O2 -target x86_64-windows-gnu tensorchat_cli.c -lwinhttp -o tensorchat_cli.exe
 *
 * Usage:
 *   tensorchat_cli.exe                          # interactive mode (localhost:8080)
 *   tensorchat_cli.exe --host 10.0.2.15:8080    # custom endpoint
 *   tensorchat_cli.exe --prompt "hello"          # single-shot prompt
 *   tensorchat_cli.exe --bench                   # print tok/s from last generation
 */

#include <windows.h>
#include <winhttp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#pragma comment(lib, "winhttp.lib")

/* ── Configuration ──────────────────────────────────────────────────── */

static wchar_t g_host[256] = L"127.0.0.1";
static INTERNET_PORT g_port = 8080;
static int g_max_tokens = 512;
static float g_temperature = 0.7f;

/* ── HTTP transport ─────────────────────────────────────────────────── */

static int http_post_json(const wchar_t *path, const char *body, int body_len,
                          char *resp, int resp_cap)
{
    HINTERNET sess = NULL, conn = NULL, req = NULL;
    DWORD bytes_read = 0, total = 0;
    DWORD status = 0, status_sz = sizeof(status);
    int ok = 0;

    sess = WinHttpOpen(L"TensorChat-CLI/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!sess) goto done;

    WinHttpSetTimeouts(sess, 5000, 5000, 30000, 60000);

    conn = WinHttpConnect(sess, g_host, g_port, 0);
    if (!conn) goto done;

    req = WinHttpOpenRequest(conn, L"POST", path, NULL,
                             WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!req) goto done;

    if (!WinHttpSendRequest(req, L"Content-Type: application/json\r\n", (DWORD)-1,
                            (LPVOID)body, (DWORD)body_len, (DWORD)body_len, 0))
        goto done;

    if (!WinHttpReceiveResponse(req, NULL)) goto done;

    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        NULL, &status, &status_sz, NULL);

    while (WinHttpReadData(req, resp + total, (DWORD)(resp_cap - total - 1), &bytes_read)
           && bytes_read > 0)
        total += bytes_read;

    resp[total] = '\0';
    ok = (status >= 200 && status < 300 && total > 0);

done:
    if (req) WinHttpCloseHandle(req);
    if (conn) WinHttpCloseHandle(conn);
    if (sess) WinHttpCloseHandle(sess);
    return ok;
}

static int http_get(const wchar_t *path, char *resp, int resp_cap)
{
    HINTERNET sess = NULL, conn = NULL, req = NULL;
    DWORD bytes_read = 0, total = 0;
    DWORD status = 0, status_sz = sizeof(status);
    int ok = 0;

    sess = WinHttpOpen(L"TensorChat-CLI/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!sess) goto done;

    WinHttpSetTimeouts(sess, 3000, 3000, 10000, 10000);

    conn = WinHttpConnect(sess, g_host, g_port, 0);
    if (!conn) goto done;

    req = WinHttpOpenRequest(conn, L"GET", path, NULL,
                             WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!req) goto done;

    if (!WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
        goto done;

    if (!WinHttpReceiveResponse(req, NULL)) goto done;

    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        NULL, &status, &status_sz, NULL);

    while (WinHttpReadData(req, resp + total, (DWORD)(resp_cap - total - 1), &bytes_read)
           && bytes_read > 0)
        total += bytes_read;

    resp[total] = '\0';
    ok = (status >= 200 && status < 300);

done:
    if (req) WinHttpCloseHandle(req);
    if (conn) WinHttpCloseHandle(conn);
    if (sess) WinHttpCloseHandle(sess);
    return ok;
}

/* ── JSON helpers ───────────────────────────────────────────────────── */

static void json_escape(const char *src, char *dst, int cap)
{
    int o = 0;
    for (int i = 0; src[i] && o + 6 < cap; i++) {
        char c = src[i];
        if (c == '\\' || c == '"') { dst[o++] = '\\'; dst[o++] = c; }
        else if (c == '\n') { dst[o++] = '\\'; dst[o++] = 'n'; }
        else if (c == '\r') { dst[o++] = '\\'; dst[o++] = 'r'; }
        else if (c == '\t') { dst[o++] = '\\'; dst[o++] = 't'; }
        else if ((unsigned char)c >= 32) dst[o++] = c;
    }
    dst[o] = '\0';
}

/* Extract a string value for a given JSON key (simple flat parser) */
static int json_extract_string(const char *json, const char *key,
                                char *out, int cap)
{
    char pat[64];
    const char *k;
    int o = 0;

    snprintf(pat, sizeof(pat), "\"%s\":\"", key);
    k = strstr(json, pat);
    if (!k) {
        /* Try with space after colon */
        snprintf(pat, sizeof(pat), "\"%s\": \"", key);
        k = strstr(json, pat);
    }
    if (!k) return 0;
    k += strlen(pat);

    while (*k && o + 1 < cap) {
        if (*k == '"') break;
        if (*k == '\\') {
            k++;
            if (!*k) break;
            if (*k == 'n') out[o++] = '\n';
            else if (*k == 'r') out[o++] = '\r';
            else if (*k == 't') out[o++] = '\t';
            else if (*k == '"') out[o++] = '"';
            else if (*k == '\\') out[o++] = '\\';
            else out[o++] = *k;
            k++;
            continue;
        }
        out[o++] = *k++;
    }
    out[o] = '\0';
    return o > 0;
}

/* Extract a number value for a given JSON key */
static int json_extract_number(const char *json, const char *key, double *val)
{
    char pat[64];
    const char *k;

    snprintf(pat, sizeof(pat), "\"%s\":", key);
    k = strstr(json, pat);
    if (!k) return 0;
    k += strlen(pat);
    while (*k == ' ') k++;
    *val = atof(k);
    return 1;
}

/* ── Chat command ───────────────────────────────────────────────────── */

static int do_chat(const char *prompt, char *answer, int cap)
{
    char esc[2048], body[4096], resp[16384];
    const wchar_t *paths[] = {
        L"/v1/chat/completions",
        L"/v1/chat",
        L"/chat/completions"
    };

    json_escape(prompt, esc, sizeof(esc));

    /* Try OpenAI-compatible chat endpoint first */
    snprintf(body, sizeof(body),
             "{\"model\":\"tensoros\",\"messages\":[{\"role\":\"user\","
             "\"content\":\"%s\"}],\"max_tokens\":%d,\"temperature\":%.2f}",
             esc, g_max_tokens, g_temperature);

    for (int i = 0; i < 3; i++) {
        const char *use_body = (i == 1)
            ? body  /* /v1/chat uses same format */
            : body;

        /* For /v1/chat, try simpler prompt format */
        if (i == 1) {
            char simple[4096];
            snprintf(simple, sizeof(simple),
                     "{\"model\":\"tensoros\",\"prompt\":\"%s\","
                     "\"max_tokens\":%d,\"temperature\":%.2f}",
                     esc, g_max_tokens, g_temperature);
            use_body = simple;
        }

        if (http_post_json(paths[i], use_body, (int)strlen(use_body),
                           resp, sizeof(resp))) {
            /* Try to extract answer from various response formats */
            if (json_extract_string(resp, "content", answer, cap)) return 1;
            if (json_extract_string(resp, "text", answer, cap)) return 1;
            if (json_extract_string(resp, "response", answer, cap)) return 1;
            if (json_extract_string(resp, "output", answer, cap)) return 1;

            /* If we got a 200 but couldn't parse, return raw */
            int len = (int)strlen(resp);
            if (len > 0 && len < cap) {
                memcpy(answer, resp, len + 1);
                return 1;
            }
        }
    }
    return 0;
}

/* ── Status/bench command ───────────────────────────────────────────── */

static void do_status(void)
{
    char resp[4096];
    double val;

    printf("\n--- TensorOS Runtime Status ---\n");

    if (http_get(L"/v1/status", resp, sizeof(resp)) ||
        http_get(L"/status", resp, sizeof(resp))) {
        /* Parse status fields */
        char model[256] = "unknown";
        json_extract_string(resp, "model", model, sizeof(model));
        printf("  Model:      %s\n", model);

        if (json_extract_number(resp, "tok_per_sec", &val))
            printf("  Throughput: %.1f tok/s\n", val);
        if (json_extract_number(resp, "prefill_ms", &val))
            printf("  Prefill:    %.0f ms\n", val);
        if (json_extract_number(resp, "vram_mb", &val))
            printf("  VRAM:       %.0f MB\n", val);
        if (json_extract_number(resp, "cache_len", &val))
            printf("  Cache pos:  %.0f\n", val);

        printf("  Raw: %s\n", resp);
    } else {
        printf("  (runtime unreachable at %ls:%d)\n", g_host, g_port);
    }
    printf("\n");
}

/* ── Help ───────────────────────────────────────────────────────────── */

static void print_help(void)
{
    printf(
        "\nCommands:\n"
        "  /status    Show model info and tok/s\n"
        "  /bench     Same as /status\n"
        "  /temp N    Set temperature (0.0-2.0)\n"
        "  /tokens N  Set max output tokens\n"
        "  /clear     Clear screen\n"
        "  /help      Show this help\n"
        "  /quit      Exit\n"
        "\nAnything else is sent as a chat prompt.\n\n"
    );
}

/* ── Banner ─────────────────────────────────────────────────────────── */

static void print_banner(void)
{
    printf(
        "\n"
        "  ╔══════════════════════════════════════════╗\n"
        "  ║         TensorChat CLI v1.0              ║\n"
        "  ║   Command-line interface for TensorOS    ║\n"
        "  ╚══════════════════════════════════════════╝\n"
        "\n"
        "  Endpoint: %ls:%d\n"
        "  Type /help for commands, or just start chatting.\n\n",
        g_host, g_port
    );
}

/* ── Argument parsing ───────────────────────────────────────────────── */

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            i++;
            /* Parse host:port */
            char *colon = strchr(argv[i], ':');
            if (colon) {
                *colon = '\0';
                MultiByteToWideChar(CP_UTF8, 0, argv[i], -1, g_host, 256);
                g_port = (INTERNET_PORT)atoi(colon + 1);
            } else {
                MultiByteToWideChar(CP_UTF8, 0, argv[i], -1, g_host, 256);
            }
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            g_port = (INTERNET_PORT)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            g_temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            g_max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            /* Single-shot mode */
            i++;
            char answer[8192];
            if (do_chat(argv[i], answer, sizeof(answer)))
                printf("%s\n", answer);
            else
                fprintf(stderr, "Error: runtime unreachable at %ls:%d\n", g_host, g_port);
            exit(0);
        } else if (strcmp(argv[i], "--bench") == 0 || strcmp(argv[i], "--status") == 0) {
            do_status();
            exit(0);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: tensorchat_cli [options]\n\n");
            printf("Options:\n");
            printf("  --host HOST:PORT   Runtime endpoint (default: 127.0.0.1:8080)\n");
            printf("  --port PORT        Runtime port\n");
            printf("  --temp TEMP        Sampling temperature (default: 0.7)\n");
            printf("  --tokens N         Max output tokens (default: 512)\n");
            printf("  --prompt \"TEXT\"     Single-shot prompt (non-interactive)\n");
            printf("  --bench            Print model status and tok/s\n");
            printf("  --help             Show this help\n");
            exit(0);
        }
    }
}

/* ── REPL ───────────────────────────────────────────────────────────── */

static void repl(void)
{
    char line[4096];
    char answer[16384];

    print_banner();

    for (;;) {
        printf("\x1b[36m>\x1b[0m ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;

        /* Strip trailing newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;

        /* Commands */
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0 ||
            strcmp(line, "/q") == 0) break;

        if (strcmp(line, "/help") == 0 || strcmp(line, "/?") == 0) {
            print_help();
            continue;
        }

        if (strcmp(line, "/status") == 0 || strcmp(line, "/bench") == 0) {
            do_status();
            continue;
        }

        if (strcmp(line, "/clear") == 0 || strcmp(line, "/cls") == 0) {
            printf("\x1b[2J\x1b[H");
            continue;
        }

        if (strncmp(line, "/temp ", 6) == 0) {
            g_temperature = (float)atof(line + 6);
            printf("  Temperature set to %.2f\n", g_temperature);
            continue;
        }

        if (strncmp(line, "/tokens ", 8) == 0) {
            g_max_tokens = atoi(line + 8);
            if (g_max_tokens < 1) g_max_tokens = 1;
            printf("  Max tokens set to %d\n", g_max_tokens);
            continue;
        }

        /* Send as chat prompt */
        printf("\x1b[90mthinking...\x1b[0m");
        fflush(stdout);

        LARGE_INTEGER freq, t0, t1;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t0);

        int ok = do_chat(line, answer, sizeof(answer));

        QueryPerformanceCounter(&t1);
        double elapsed_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        /* Clear "thinking..." */
        printf("\r\x1b[K");

        if (ok) {
            printf("\x1b[32m%s\x1b[0m\n", answer);
            printf("\x1b[90m  (%.0f ms)\x1b[0m\n\n", elapsed_ms);
        } else {
            printf("\x1b[31mError: runtime unreachable at %ls:%d\x1b[0m\n\n",
                   g_host, g_port);
        }
    }

    printf("\nBye.\n");
}

/* ── Entry point ────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    /* Enable ANSI escape sequences on Windows 10+ */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */);

    SetConsoleOutputCP(65001);  /* UTF-8 */

    parse_args(argc, argv);
    repl();
    return 0;
}
