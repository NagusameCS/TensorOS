#include <windows.h>
#include <commctrl.h>
#include <winhttp.h>
#include <wchar.h>

#pragma comment(lib, "winhttp.lib")

#define APP_CLASS_NAME L"TensorChatNativeWindow"
#define BOOT_TIMER_ID 77

#define IDC_BOOT_TEXT      100
#define IDC_BOOT_PROGRESS  101

#define IDC_ADD_CHAT       200
#define IDC_ADD_EDITOR     201
#define IDC_ADD_TERMINAL   202
#define IDC_ADD_OUTLINE    203
#define IDC_THEME_COMBO    204
#define IDC_STATUS         205

#define IDC_CHAT_LOG       300
#define IDC_CHAT_INPUT     301
#define IDC_CHAT_SEND      302

#define IDC_EDITOR         400
#define IDC_OUTLINE        401

#define IDC_TERM_LOG       500
#define IDC_TERM_INPUT     501
#define IDC_TERM_RUN       502

/* panel header section labels */
#define IDC_LBL_CHAT_HDR   600
#define IDC_LBL_EDIT_HDR   601
#define IDC_LBL_OUTL_HDR   602
#define IDC_LBL_TERM_HDR   603
#define IDC_APP_TITLE      610
#define IDC_APP_SUBTITLE   611

typedef struct {
    const wchar_t *name;
    COLORREF bg;        /* window bg            */
    COLORREF panel;     /* panel / card bg      */
    COLORREF header;    /* section header strip */
    COLORREF input;     /* edit control bg      */
    COLORREF text;      /* primary text         */
    COLORREF text_dim;  /* secondary / muted    */
    COLORREF accent;    /* accent / focus       */
    COLORREF border;    /* 1-px border          */
    COLORREF btn_bg;    /* button face          */
    COLORREF btn_text;  /* button label         */
} theme_t;

#define C RGB
static const theme_t g_themes[] = {
/* name                    bg              panel           header          input           text            text_dim        accent          border          btn_bg          btn_text       */
{L"Nagusame",             C(0,0,0),       C(10,10,10),    C(20,20,20),    C(10,10,10),    C(255,255,255), C(153,153,153), C(255,255,255), C(34,34,34),    C(20,20,20),    C(255,255,255)},
{L"VS Dark+",             C(30,30,30),    C(37,37,38),    C(45,45,48),    C(30,30,30),    C(212,212,212), C(157,165,180), C(86,156,214),  C(62,62,62),    C(45,45,48),    C(212,212,212)},
{L"VS Light+",            C(255,255,255), C(243,243,243), C(232,232,232), C(255,255,255), C(0,0,0),       C(113,113,113), C(0,95,184),    C(200,200,200), C(243,243,243), C(0,0,0)      },
{L"Monokai",              C(39,40,34),    C(30,31,28),    C(45,46,40),    C(39,40,34),    C(248,248,242), C(117,113,94),  C(166,226,46),  C(62,61,50),    C(62,61,50),    C(248,248,242)},
{L"Abyss",                C(0,12,24),     C(5,24,38),     C(10,31,50),    C(5,24,38),     C(102,136,204), C(56,72,135),   C(34,170,153),  C(21,48,69),    C(14,37,59),    C(102,136,204)},
{L"One Dark Pro",         C(40,44,52),    C(33,37,43),    C(44,50,58),    C(29,32,38),    C(171,178,191), C(92,99,112),   C(97,175,239),  C(62,68,82),    C(44,50,58),    C(171,178,191)},
{L"Solarized Dark",       C(0,43,54),     C(7,54,66),     C(10,68,84),    C(0,43,54),     C(131,148,150), C(88,110,117),  C(38,139,210),  C(22,75,90),    C(7,54,66),     C(131,148,150)},
{L"Solarized Light",      C(253,246,227), C(238,232,213), C(229,223,201), C(253,246,227), C(101,123,131), C(147,161,161), C(38,139,210),  C(207,200,184), C(238,232,213), C(101,123,131)},
{L"Tomorrow Night Blue",  C(0,36,81),     C(0,43,96),     C(0,52,110),    C(0,43,96),     C(255,255,255), C(187,218,255), C(255,204,102), C(0,63,138),    C(0,64,128),    C(255,255,255)},
{L"GitHub Dark",          C(13,17,23),    C(22,27,34),    C(28,33,40),    C(22,27,34),    C(201,209,217), C(139,148,158), C(88,166,255),  C(48,54,61),    C(33,38,45),    C(201,209,217)},
{L"GitHub Light",         C(255,255,255), C(246,248,250), C(234,238,242), C(255,255,255), C(36,41,47),    C(87,96,106),   C(9,105,218),   C(208,215,222), C(246,248,250), C(36,41,47)   },
{L"Nord",                 C(46,52,64),    C(59,66,82),    C(67,76,94),    C(59,66,82),    C(216,222,233), C(160,168,185), C(136,192,208), C(76,86,106),   C(67,76,94),    C(216,222,233)},
{L"Dracula",              C(40,42,54),    C(33,34,44),    C(52,55,70),    C(40,42,54),    C(248,248,242), C(98,114,164),  C(189,147,249), C(68,71,90),    C(52,55,70),    C(248,248,242)},
{L"Catppuccin Mocha",     C(30,30,46),    C(24,24,37),    C(49,50,68),    C(24,24,37),    C(205,214,244), C(166,173,200), C(203,166,247), C(69,71,90),    C(49,50,68),    C(205,214,244)},
{L"Catppuccin Latte",     C(239,241,245), C(230,233,239), C(220,224,232), C(239,241,245), C(76,79,105),   C(140,143,161), C(136,57,239),  C(204,208,218), C(230,233,239), C(76,79,105)  },
{L"Kimbie Dark",          C(34,26,15),    C(45,33,24),    C(54,42,30),    C(34,26,15),    C(211,175,134), C(165,122,74),  C(220,57,88),   C(74,55,40),    C(54,42,30),    C(211,175,134)},
{L"Quiet Light",          C(245,245,245), C(232,232,232), C(220,220,220), C(255,255,255), C(51,51,51),    C(119,119,119), C(64,120,242),  C(204,204,204), C(232,232,232), C(51,51,51)   },
{L"IJ Darcula",           C(43,43,43),    C(49,51,53),    C(60,63,65),    C(69,73,74),    C(169,183,198), C(96,99,102),   C(94,159,212),  C(75,77,78),    C(76,80,82),    C(187,187,187)},
{L"IJ Light",             C(255,255,255), C(242,242,242), C(232,232,232), C(255,255,255), C(0,0,0),       C(120,120,120), C(38,117,191),  C(209,209,209), C(242,242,242), C(0,0,0)      },
{L"IJ New UI Dark",       C(30,31,34),    C(39,40,44),    C(43,45,48),    C(30,31,34),    C(223,225,229), C(128,128,128), C(79,154,249),  C(61,63,65),    C(74,75,77),    C(223,225,229)},
{L"HC Dark",              C(0,0,0),       C(13,13,13),    C(26,26,26),    C(0,0,0),       C(255,255,0),   C(192,192,192), C(0,255,0),     C(255,255,255), C(26,26,26),    C(255,255,0)  },
{L"HC Light",             C(255,255,255), C(240,240,240), C(224,224,224), C(255,255,255), C(0,0,0),       C(51,51,51),    C(0,0,255),     C(0,0,0),       C(224,224,224), C(0,0,0)      },
};
#undef C

#define THEME_COUNT ((int)(sizeof(g_themes)/sizeof(g_themes[0])))

static int g_theme_idx = 1;   /* default: VS Dark+ */

/* ── brushes ─────────────────────────────────────────────────────────────── */
static HBRUSH g_br_bg     = NULL;
static HBRUSH g_br_panel  = NULL;
static HBRUSH g_br_header = NULL;
static HBRUSH g_br_input  = NULL;
static HBRUSH g_br_btn    = NULL;

/* ── fonts ────────────────────────────────────────────────────────────────── */
static HFONT g_font_ui      = NULL;   /* Segoe UI 13 normal          */
static HFONT g_font_ui_bold = NULL;   /* Segoe UI 12 semibold labels */
static HFONT g_font_mono    = NULL;   /* Cascadia Code / Consolas 13 */
static HFONT g_font_title   = NULL;   /* boot screen title           */

static HWND g_boot_text = NULL;
static HWND g_boot_progress = NULL;

static HWND g_btn_add_chat = NULL;
static HWND g_btn_add_editor = NULL;
static HWND g_btn_add_terminal = NULL;
static HWND g_btn_add_outline = NULL;
static HWND g_theme_combo = NULL;
static HWND g_status = NULL;
static HWND g_app_title = NULL;
static HWND g_app_subtitle = NULL;

static HWND g_chat_log = NULL;
static HWND g_chat_input = NULL;
static HWND g_chat_send = NULL;

static HWND g_editor = NULL;
static HWND g_outline = NULL;

static HWND g_term_log = NULL;
static HWND g_term_input = NULL;
static HWND g_term_run = NULL;

static WNDPROC g_chat_input_oldproc = NULL;
static WNDPROC g_term_input_oldproc = NULL;

static int g_chat_waiting = 0;
static wchar_t g_last_runtime_error[256] = L"";

/* Main window handle — set in WM_CREATE so worker threads can PostMessage back */
static HWND g_main_hwnd = NULL;

/* Async message IDs posted from worker threads to the main window */
#define WM_APP_CHAT_REPLY (WM_APP + 1)
#define WM_APP_TERM_DONE  (WM_APP + 2)

/* Per-request heap context for chat worker thread */
typedef struct {
    HWND    hwnd;
    wchar_t prompt[1024];
    wchar_t reply[2048];
    int     ok;
} chat_async_ctx_t;

/* Per-request heap context for terminal worker thread */
typedef struct {
    HWND    hwnd;
    wchar_t cmd[512];
    wchar_t output[4096];
} term_async_ctx_t;

/* section header label controls */
static HWND g_lbl_chat   = NULL;
static HWND g_lbl_editor = NULL;
static HWND g_lbl_outline = NULL;
static HWND g_lbl_term   = NULL;

static int g_boot_done = 0;
static int g_show_chat = 1;
static int g_show_editor = 1;
static int g_show_outline = 0;
static int g_show_terminal = 0;

static int visible_module_count(void)
{
    return (g_show_chat ? 1 : 0) + (g_show_editor ? 1 : 0) +
           (g_show_outline ? 1 : 0) + (g_show_terminal ? 1 : 0);
}

static void update_module_button_labels(void)
{
    if (g_btn_add_chat)     SetWindowTextW(g_btn_add_chat,     g_show_chat ? L"Chat On" : L"Chat Off");
    if (g_btn_add_editor)   SetWindowTextW(g_btn_add_editor,   g_show_editor ? L"Editor On" : L"Editor Off");
    if (g_btn_add_terminal) SetWindowTextW(g_btn_add_terminal, g_show_terminal ? L"Terminal On" : L"Terminal Off");
    if (g_btn_add_outline)  SetWindowTextW(g_btn_add_outline,  g_show_outline ? L"Outline On" : L"Outline Off");
}

static void recreate_brushes(void)
{
    const theme_t *t = &g_themes[g_theme_idx];
#define DEL(b) do { if (b) { DeleteObject(b); (b) = NULL; } } while(0)
    DEL(g_br_bg); DEL(g_br_panel); DEL(g_br_header);
    DEL(g_br_input); DEL(g_br_btn);
#undef DEL
    g_br_bg     = CreateSolidBrush(t->bg);
    g_br_panel  = CreateSolidBrush(t->panel);
    g_br_header = CreateSolidBrush(t->header);
    g_br_input  = CreateSolidBrush(t->input);
    g_br_btn    = CreateSolidBrush(t->btn_bg);
}

static void recreate_fonts(void)
{
#define DEL(f) do { if (f) { DeleteObject(f); (f) = NULL; } } while(0)
    DEL(g_font_ui); DEL(g_font_ui_bold); DEL(g_font_mono); DEL(g_font_title);
#undef DEL
    g_font_ui      = CreateFontW(-13, 0, 0, 0, FW_NORMAL,   0,0,0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                  CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH|FF_SWISS, L"Segoe UI");
    g_font_ui_bold = CreateFontW(-12, 0, 0, 0, FW_SEMIBOLD, 0,0,0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                  CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH|FF_SWISS, L"Segoe UI");
    g_font_mono    = CreateFontW(-14, 0, 0, 0, FW_NORMAL,   0,0,0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                  CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH|FF_MODERN, L"Cascadia Code");
    if (!g_font_mono)
        g_font_mono = CreateFontW(-14, 0, 0, 0, FW_NORMAL,  0,0,0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                  CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH|FF_MODERN, L"Consolas");
    g_font_title   = CreateFontW(-20, 0, 0, 0, FW_SEMIBOLD, 0,0,0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                  CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH|FF_SWISS, L"Segoe UI");
}

/* strip visual styles → flat 1-px WS_BORDER on dark themes */
static void strip_ctrl_theme(HWND h)
{
    typedef HRESULT (WINAPI *pfnSWT)(HWND, LPCWSTR, LPCWSTR);
    static pfnSWT fn = (pfnSWT)(void*)-1;
    if (fn == (pfnSWT)(void*)-1) {
        HMODULE m = GetModuleHandleW(L"uxtheme.dll");
        if (!m) m = LoadLibraryW(L"uxtheme.dll");
        fn = m ? (pfnSWT)GetProcAddress(m, "SetWindowTheme") : NULL;
    }
    if (fn && h) fn(h, L"", L"");
}

/* dark title bar via DWMAPI (Windows 10 1809+) */
static void set_title_dark(HWND hwnd, BOOL dark)
{
    typedef HRESULT (WINAPI *pfnDWA)(HWND, DWORD, LPCVOID, DWORD);
    static pfnDWA fn = (pfnDWA)(void*)-1;
    if (fn == (pfnDWA)(void*)-1) {
        HMODULE m = LoadLibraryW(L"dwmapi.dll");
        fn = m ? (pfnDWA)GetProcAddress(m, "DwmSetWindowAttribute") : NULL;
    }
    if (!fn) return;
    fn(hwnd, 20, &dark, sizeof(BOOL));  /* DWMWA_USE_IMMERSIVE_DARK_MODE */
    fn(hwnd, 19, &dark, sizeof(BOOL));  /* older pre-release attribute   */
}

static BOOL theme_is_dark(int idx)
{
    COLORREF bg = g_themes[idx].bg;
    int lum = (GetRValue(bg)*299 + GetGValue(bg)*587 + GetBValue(bg)*114) / 1000;
    return lum < 128;
}

/* flat owner-draw button renderer — called from WM_DRAWITEM */
static void draw_button(LPDRAWITEMSTRUCT dis)
{
    const theme_t *t = &g_themes[g_theme_idx];
    BOOL pressed = (dis->itemState & ODS_SELECTED) != 0;
    BOOL focused = (dis->itemState & ODS_FOCUS)    != 0;
    int id = GetDlgCtrlID(dis->hwndItem);
    BOOL module_on = FALSE;
    COLORREF face = pressed ? t->accent   : t->btn_bg;
    COLORREF fc   = pressed ? t->bg       : t->btn_text;
    RECT r = dis->rcItem;
    wchar_t txt[128];
    HBRUSH br;
    HPEN pen, old_pen;
    HFONT old_font;

    GetWindowTextW(dis->hwndItem, txt, 128);

    if (id == IDC_ADD_CHAT) module_on = g_show_chat;
    else if (id == IDC_ADD_EDITOR) module_on = g_show_editor;
    else if (id == IDC_ADD_TERMINAL) module_on = g_show_terminal;
    else if (id == IDC_ADD_OUTLINE) module_on = g_show_outline;

    if (module_on && !pressed) {
        face = t->accent;
        fc = t->bg;
    }

    /* fill */
    br = CreateSolidBrush(face);
    FillRect(dis->hDC, &r, br);
    DeleteObject(br);

    /* 1-px border */
    pen = CreatePen(PS_SOLID, 1, t->border);
    old_pen = (HPEN)SelectObject(dis->hDC, pen);
    MoveToEx(dis->hDC, r.left,     r.top,      NULL);
    LineTo  (dis->hDC, r.right-1,  r.top);
    LineTo  (dis->hDC, r.right-1,  r.bottom-1);
    LineTo  (dis->hDC, r.left,     r.bottom-1);
    LineTo  (dis->hDC, r.left,     r.top);
    SelectObject(dis->hDC, old_pen);
    DeleteObject(pen);

    /* accent inner focus ring */
    if (focused) {
        RECT ir = { r.left+2, r.top+2, r.right-2, r.bottom-2 };
        HPEN fp = CreatePen(PS_DOT, 1, t->accent);
        HPEN op = (HPEN)SelectObject(dis->hDC, fp);
        SetBkMode(dis->hDC, TRANSPARENT);
        MoveToEx(dis->hDC, ir.left,   ir.top,      NULL);
        LineTo  (dis->hDC, ir.right,  ir.top);
        LineTo  (dis->hDC, ir.right,  ir.bottom);
        LineTo  (dis->hDC, ir.left,   ir.bottom);
        LineTo  (dis->hDC, ir.left,   ir.top);
        SelectObject(dis->hDC, op);
        DeleteObject(fp);
    }

    /* label */
    SetBkMode(dis->hDC, TRANSPARENT);
    SetTextColor(dis->hDC, fc);
    old_font = (HFONT)SelectObject(dis->hDC, g_font_ui ? g_font_ui : (HFONT)GetStockObject(DEFAULT_GUI_FONT));
    DrawTextW(dis->hDC, txt, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOCLIP);
    SelectObject(dis->hDC, old_font);
}

static void append_text(HWND hEdit, const wchar_t *text)
{
    int len = GetWindowTextLengthW(hEdit);
    SendMessageW(hEdit, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(hEdit, EM_REPLACESEL, FALSE, (LPARAM)text);
}

static int is_ws(wchar_t c)
{
    return c == L' ' || c == L'\n' || c == L'\r' || c == L'\t';
}

static void trim_copy(const wchar_t *src, wchar_t *dst, int cap)
{
    int n = (int)wcslen(src);
    int a = 0, b = n;
    while (a < n && is_ws(src[a])) a++;
    while (b > a && is_ws(src[b - 1])) b--;
    {
        int out = 0;
        int i;
        for (i = a; i < b && out + 1 < cap; i++) dst[out++] = src[i];
        dst[out] = L'\0';
    }
}

static void set_status(const wchar_t *s)
{
    if (g_status) SetWindowTextW(g_status, s);
}

static void current_time_tag(wchar_t *out, int cap)
{
    SYSTEMTIME st;
    GetLocalTime(&st);
    wsprintfW(out, L"[%02d:%02d]", st.wHour, st.wMinute);
    out[cap - 1] = L'\0';
}

static void json_escape_wide(const wchar_t *src, wchar_t *dst, int cap)
{
    int out = 0;
    int i;
    for (i = 0; src[i] && out + 2 < cap; i++) {
        wchar_t c = src[i];
        if (c == L'\\' || c == L'"') {
            if (out + 2 >= cap) break;
            dst[out++] = L'\\';
            dst[out++] = c;
        } else if (c == L'\n') {
            if (out + 2 >= cap) break;
            dst[out++] = L'\\';
            dst[out++] = L'n';
        } else if (c == L'\r') {
            if (out + 2 >= cap) break;
            dst[out++] = L'\\';
            dst[out++] = L'r';
        } else if (c == L'\t') {
            if (out + 2 >= cap) break;
            dst[out++] = L'\\';
            dst[out++] = L't';
        } else if (c >= 32) {
            dst[out++] = c;
        }
    }
    dst[out] = L'\0';
}

static int extract_json_string_key(const wchar_t *json, const wchar_t *key, wchar_t *out, int cap)
{
    wchar_t pat[64];
    const wchar_t *k;
    int outp = 0;

    wsprintfW(pat, L"\"%s\":\"", key);
    k = wcsstr(json, pat);
    if (!k) return 0;
    k += wcslen(pat);

    while (*k && outp + 1 < cap) {
        if (*k == L'"') break;
        if (*k == L'\\') {
            k++;
            if (!*k) break;
            if (*k == L'n') out[outp++] = L'\n';
            else if (*k == L'r') out[outp++] = L'\r';
            else if (*k == L't') out[outp++] = L'\t';
            else if (*k == L'"') out[outp++] = L'"';
            else if (*k == L'\\') out[outp++] = L'\\';
            else out[outp++] = *k;
            k++;
            continue;
        }
        out[outp++] = *k++;
    }
    out[outp] = L'\0';
    return outp;
}

static int extract_runtime_answer(const wchar_t *json, wchar_t *out, int cap)
{
    if (extract_json_string_key(json, L"content", out, cap) > 0) return 1;
    if (extract_json_string_key(json, L"response", out, cap) > 0) return 1;
    if (extract_json_string_key(json, L"text", out, cap) > 0) return 1;
    if (extract_json_string_key(json, L"answer", out, cap) > 0) return 1;
    return 0;
}

static int runtime_chat_request_path(const wchar_t *path,
                                     const wchar_t *body_wide,
                                     wchar_t *out,
                                     int cap,
                                     wchar_t *diag,
                                     int diag_cap)
{
    HINTERNET sess = NULL, conn = NULL, req = NULL;
    int body_utf8_len, resp_ok = 0;
    DWORD status = 0, status_len = sizeof(status);
    static const wchar_t *host = L"127.0.0.1";
    (void)diag_cap;

    char body_utf8[6144];
    char resp_utf8[16384];
    DWORD avail = 0, read_n = 0;
    int resp_len = 0;

    body_utf8_len = WideCharToMultiByte(CP_UTF8, 0, body_wide, -1, body_utf8,
                                        (int)sizeof(body_utf8), NULL, NULL);
    if (body_utf8_len <= 1) {
        wsprintfW(diag, L"request body conversion failed");
        return 0;
    }

    sess = WinHttpOpen(L"TensorChatNative/1.0",
                       WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                       WINHTTP_NO_PROXY_NAME,
                       WINHTTP_NO_PROXY_BYPASS,
                       0);
    if (!sess) goto cleanup;

    WinHttpSetTimeouts(sess, 2000, 2000, 6000, 10000);

    conn = WinHttpConnect(sess, host, 8080, 0);
    if (!conn) goto cleanup;

    req = WinHttpOpenRequest(conn,
                             L"POST",
                             path,
                             NULL,
                             WINHTTP_NO_REFERER,
                             WINHTTP_DEFAULT_ACCEPT_TYPES,
                             0);
    if (!req) goto cleanup;

    {
        const wchar_t *hdr = L"Content-Type: application/json\r\nAccept: application/json\r\n";
        if (!WinHttpSendRequest(req,
                                hdr,
                                (DWORD)-1,
                                body_utf8,
                                (DWORD)(body_utf8_len - 1),
                                (DWORD)(body_utf8_len - 1),
                                0)) {
            goto cleanup;
        }
    }

    if (!WinHttpReceiveResponse(req, NULL)) goto cleanup;

    WinHttpQueryHeaders(req,
                        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX,
                        &status,
                        &status_len,
                        WINHTTP_NO_HEADER_INDEX);

    while (WinHttpQueryDataAvailable(req, &avail) && avail > 0) {
        DWORD want = avail;
        if (want > (DWORD)(sizeof(resp_utf8) - 1 - resp_len)) {
            want = (DWORD)(sizeof(resp_utf8) - 1 - resp_len);
        }
        if (want == 0) break;
        if (!WinHttpReadData(req, resp_utf8 + resp_len, want, &read_n)) break;
        if (read_n == 0) break;
        resp_len += (int)read_n;
        if (resp_len >= (int)sizeof(resp_utf8) - 1) break;
    }
    resp_utf8[resp_len] = '\0';

    if (resp_len > 0) {
        wchar_t resp_w[16384];
        int n = MultiByteToWideChar(CP_UTF8, 0, resp_utf8, -1, resp_w,
                                    (int)(sizeof(resp_w) / sizeof(resp_w[0])));
        if (n > 0) {
            if (status >= 200 && status < 300 && extract_runtime_answer(resp_w, out, cap)) {
                resp_ok = 1;
            } else if (status >= 200 && status < 300) {
                wsprintfW(diag, L"HTTP %lu but no known answer field", (unsigned long)status);
            } else {
                wsprintfW(diag, L"HTTP %lu", (unsigned long)status);
            }
        } else {
            wsprintfW(diag, L"UTF-8 decode failed");
        }
    } else {
        if (status > 0) wsprintfW(diag, L"HTTP %lu with empty body", (unsigned long)status);
        else wsprintfW(diag, L"empty runtime response");
    }

cleanup:
    if (!resp_ok && diag[0] == L'\0') {
        DWORD e = GetLastError();
        if (e) wsprintfW(diag, L"WinHTTP error %lu", (unsigned long)e);
        else wsprintfW(diag, L"runtime connection failed");
    }
    if (req) WinHttpCloseHandle(req);
    if (conn) WinHttpCloseHandle(conn);
    if (sess) WinHttpCloseHandle(sess);
    return resp_ok;
}

static int runtime_chat_request(const wchar_t *prompt, wchar_t *out, int cap)
{
    wchar_t esc_prompt[2048];
    wchar_t body_chat[3072];
    wchar_t body_prompt[3072];
    wchar_t diag[256] = L"";

    json_escape_wide(prompt, esc_prompt, (int)(sizeof(esc_prompt) / sizeof(esc_prompt[0])));
    wsprintfW(body_chat,
              L"{\"model\":\"tensoros\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"max_tokens\":192}",
              esc_prompt);
    wsprintfW(body_prompt,
              L"{\"model\":\"tensoros\",\"prompt\":\"%s\",\"max_tokens\":192}",
              esc_prompt);

    if (runtime_chat_request_path(L"/v1/chat/completions", body_chat, out, cap, diag, 256)) {
        g_last_runtime_error[0] = L'\0';
        return 1;
    }
    if (runtime_chat_request_path(L"/v1/chat", body_prompt, out, cap, diag, 256)) {
        g_last_runtime_error[0] = L'\0';
        return 1;
    }
    if (runtime_chat_request_path(L"/chat/completions", body_chat, out, cap, diag, 256)) {
        g_last_runtime_error[0] = L'\0';
        return 1;
    }

    wcsncpy(g_last_runtime_error, diag, (sizeof(g_last_runtime_error) / sizeof(g_last_runtime_error[0])) - 1);
    g_last_runtime_error[(sizeof(g_last_runtime_error) / sizeof(g_last_runtime_error[0])) - 1] = L'\0';
    return 0;
}

static void update_outline(void)
{
    wchar_t buf[32768];
    wchar_t line[512];
    int i = 0, li = 0, line_no = 1;
    int n;

    if (!g_editor || !g_outline) return;
    SendMessageW(g_outline, LB_RESETCONTENT, 0, 0);

    n = GetWindowTextW(g_editor, buf, (int)(sizeof(buf) / sizeof(buf[0])));
    if (n <= 0) return;

    for (i = 0; i <= n; i++) {
        wchar_t c = buf[i];
        if (c == L'\r') continue;
        if (c == L'\n' || c == L'\0') {
            line[li] = L'\0';
            if (wcsncmp(line, L"int ", 4) == 0 ||
                wcsncmp(line, L"void ", 5) == 0 ||
                wcsncmp(line, L"class ", 6) == 0 ||
                wcsncmp(line, L"struct ", 7) == 0 ||
                wcsncmp(line, L"fn ", 3) == 0 ||
                wcsncmp(line, L"def ", 4) == 0) {
                wchar_t item[620];
                wsprintfW(item, L"L%-4d %s", line_no, line);
                SendMessageW(g_outline, LB_ADDSTRING, 0, (LPARAM)item);
            }
            li = 0;
            line_no++;
        } else if (li + 1 < (int)(sizeof(line) / sizeof(line[0]))) {
            line[li++] = c;
        }
    }
}

static void chat_reply(const wchar_t *input, wchar_t *out, int cap)
{
    wchar_t lower[1024];
    int i;

    for (i = 0; i < 1023 && input[i]; i++) {
        wchar_t c = input[i];
        if (c >= L'A' && c <= L'Z') c = (wchar_t)(c - L'A' + L'a');
        lower[i] = c;
    }
    lower[i] = L'\0';

    if (wcscmp(lower, L"alive") == 0) {
        wsprintfW(out, L"Yes, I am alive. Native UI loop and chat module are running.");
        return;
    }
    if (wcscmp(lower, L"help") == 0) {
        wsprintfW(out, L"Try: 'alive', 'modules', 'theme', or ask for a coding task.\r\nChat tries runtime endpoints first, then local fallback.");
        return;
    }
    if (wcscmp(lower, L"modules") == 0) {
        wsprintfW(out, L"Modules available: chat, editor, outline, terminal. Chat is default; add others from the top bar.");
        return;
    }
    if (wcscmp(lower, L"theme") == 0) {
        wsprintfW(out, L"Theme catalog loaded: Nagusame + VS Code + IntelliJ + accessibility themes.");
        return;
    }

    if (runtime_chat_request(input, out, cap)) {
        return;
    }

    {
        int code_len = g_editor ? GetWindowTextLengthW(g_editor) : 0;
        wsprintfW(out,
                  L"Runtime unavailable. Local fallback active (%s). Prompt length=%d chars, editor buffer=%d chars.",
                  g_last_runtime_error[0] ? g_last_runtime_error : L"no details",
                  (int)wcslen(input),
                  code_len);
    }
}

/* ---------------------------------------------------------------------------
 * Async helpers: run_cmd_capture, chat_async_thread, term_async_thread
 * These run on background threads and post WM_APP_CHAT_REPLY / WM_APP_TERM_DONE
 * back to g_main_hwnd when done.
 * -------------------------------------------------------------------------*/

/* Execute cmd.exe /C <cmd>, capture stdout+stderr into out_buf (wide, OEM→UTF-16).
 * Blocks the calling thread for up to 10 s then kills the child process. */
static void run_cmd_capture(const wchar_t *cmd, wchar_t *out_buf, int out_cap)
{
    SECURITY_ATTRIBUTES sa;
    HANDLE hReadRaw = NULL, hWriteRaw = NULL;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    wchar_t cmdline[640];
    char raw[8192];
    DWORD nr = 0, total = 0;

    out_buf[0] = L'\0';
    if (out_cap < 2) return;

    sa.nLength              = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle       = TRUE;

    if (!CreatePipe(&hReadRaw, &hWriteRaw, &sa, 0)) {
        wsprintfW(out_buf, L"(pipe failed: %lu)\r\n", GetLastError());
        return;
    }
    /* Read end must not be inherited by child */
    SetHandleInformation(hReadRaw, HANDLE_FLAG_INHERIT, 0);

    ZeroMemory(&si, sizeof(si));
    si.cb          = sizeof(si);
    si.hStdOutput  = hWriteRaw;
    si.hStdError   = hWriteRaw;
    si.hStdInput   = NULL;
    si.dwFlags     = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    ZeroMemory(&pi, sizeof(pi));

    /* Build command line: cmd.exe /C <user command> */
    wsprintfW(cmdline, L"cmd.exe /C %s", cmd);
    cmdline[639] = L'\0';

    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE,
                        CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        DWORD err = GetLastError();
        CloseHandle(hWriteRaw);
        CloseHandle(hReadRaw);
        wsprintfW(out_buf, L"(launch failed: %lu)\r\n", err);
        return;
    }
    /* Close the parent's write handle so ReadFile ends when the child exits */
    CloseHandle(hWriteRaw);

    /* Wait up to 10 s, then forcibly terminate to avoid infinite hang */
    if (WaitForSingleObject(pi.hProcess, 10000) == WAIT_TIMEOUT)
        TerminateProcess(pi.hProcess, 1);

    /* Drain remaining output from the pipe */
    while (total < sizeof(raw) - 1) {
        if (!ReadFile(hReadRaw, raw + total,
                      (DWORD)(sizeof(raw) - 1 - total), &nr, NULL) || nr == 0)
            break;
        total += nr;
    }
    raw[total] = '\0';

    if (total > 0)
        MultiByteToWideChar(CP_OEMCP, 0, raw, (int)total + 1, out_buf, out_cap);
    else
        wcsncpy(out_buf, L"(no output)", out_cap - 1);
    out_buf[out_cap - 1] = L'\0';

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hReadRaw);
}

/* Worker thread: runs chat_reply (includes local cmds + runtime) off UI thread */
static DWORD WINAPI chat_async_thread(LPVOID param)
{
    chat_async_ctx_t *ctx = (chat_async_ctx_t *)param;
    chat_reply(ctx->prompt, ctx->reply,
               (int)(sizeof(ctx->reply) / sizeof(ctx->reply[0])));
    ctx->ok = !g_last_runtime_error[0];
    PostMessageW(ctx->hwnd, WM_APP_CHAT_REPLY, 0, (LPARAM)ctx);
    return 0;
}

/* Worker thread: runs shell command and captures output */
static DWORD WINAPI term_async_thread(LPVOID param)
{
    term_async_ctx_t *ctx = (term_async_ctx_t *)param;
    run_cmd_capture(ctx->cmd, ctx->output,
                    (int)(sizeof(ctx->output) / sizeof(ctx->output[0])));
    PostMessageW(ctx->hwnd, WM_APP_TERM_DONE, 0, (LPARAM)ctx);
    return 0;
}

static void on_chat_send(void)
{
    wchar_t raw[1024];
    wchar_t input[1024];
    wchar_t user_msg[1400];
    wchar_t t[16];
    chat_async_ctx_t *ctx;
    HANDLE th;

    if (g_chat_waiting) return;

    GetWindowTextW(g_chat_input, raw, (int)(sizeof(raw) / sizeof(raw[0])));
    trim_copy(raw, input, (int)(sizeof(input) / sizeof(input[0])));
    if (!input[0]) return;

    current_time_tag(t, (int)(sizeof(t) / sizeof(t[0])));
    wsprintfW(user_msg, L"%s You: %s\r\n", t, input);
    append_text(g_chat_log, user_msg);

    ctx = (chat_async_ctx_t *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*ctx));
    if (!ctx) {
        append_text(g_chat_log, L"TensorChat: (out of memory)\r\n\r\n");
        return;
    }
    ctx->hwnd = g_main_hwnd;
    wcsncpy(ctx->prompt, input, (int)(sizeof(ctx->prompt) / sizeof(ctx->prompt[0])) - 1);

    g_chat_waiting = 1;
    EnableWindow(g_chat_send, FALSE);
    set_status(L"Contacting runtime...");
    SetWindowTextW(g_chat_input, L"");
    SetFocus(g_chat_input);

    th = CreateThread(NULL, 0, chat_async_thread, ctx, 0, NULL);
    if (th) {
        CloseHandle(th);
    } else {
        /* Thread creation failed — run synchronously as fallback */
        chat_reply(ctx->prompt, ctx->reply,
                   (int)(sizeof(ctx->reply) / sizeof(ctx->reply[0])));
        ctx->ok = !g_last_runtime_error[0];
        PostMessageW(g_main_hwnd, WM_APP_CHAT_REPLY, 0, (LPARAM)ctx);
    }
}

static void on_term_run(void)
{
    wchar_t raw[512];
    wchar_t cmd[512];
    wchar_t line[640];

    GetWindowTextW(g_term_input, raw, (int)(sizeof(raw) / sizeof(raw[0])));
    trim_copy(raw, cmd, (int)(sizeof(cmd) / sizeof(cmd[0])));
    if (!cmd[0]) return;

    wsprintfW(line, L"$ %s\r\n", cmd);
    append_text(g_term_log, line);

    if (wcscmp(cmd, L"help") == 0) {
        append_text(g_term_log,
                    L"commands: help, clear, pwd, status, or any shell command\r\n\r\n");
    } else if (wcscmp(cmd, L"clear") == 0) {
        SetWindowTextW(g_term_log, L"");
    } else if (wcscmp(cmd, L"pwd") == 0) {
        wchar_t cwd[MAX_PATH];
        GetCurrentDirectoryW(MAX_PATH, cwd);
        wsprintfW(line, L"%s\r\n\r\n", cwd);
        append_text(g_term_log, line);
    } else if (wcscmp(cmd, L"status") == 0) {
        append_text(g_term_log,
                    L"terminal module active (cmd.exe passthrough via CreateProcess)\r\n\r\n");
    } else {
        /* Execute via cmd.exe in a background thread */
        term_async_ctx_t *ctx = (term_async_ctx_t *)HeapAlloc(
            GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*ctx));
        if (!ctx) {
            append_text(g_term_log, L"(out of memory)\r\n\r\n");
        } else {
            HANDLE th;
            ctx->hwnd = g_main_hwnd;
            wcsncpy(ctx->cmd, cmd,
                    (int)(sizeof(ctx->cmd) / sizeof(ctx->cmd[0])) - 1);
            append_text(g_term_log, L"(running...)\r\n");
            th = CreateThread(NULL, 0, term_async_thread, ctx, 0, NULL);
            if (th) {
                CloseHandle(th);
            } else {
                run_cmd_capture(ctx->cmd, ctx->output,
                                (int)(sizeof(ctx->output) / sizeof(ctx->output[0])));
                PostMessageW(g_main_hwnd, WM_APP_TERM_DONE, 0, (LPARAM)ctx);
            }
        }
    }

    SetWindowTextW(g_term_input, L"");
}

static LRESULT CALLBACK input_edit_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_KEYDOWN && wParam == VK_RETURN) {
        HWND parent = GetParent(hwnd);
        int id = GetDlgCtrlID(hwnd);
        if (id == IDC_CHAT_INPUT && !g_chat_waiting) {
            PostMessageW(parent, WM_COMMAND, MAKEWPARAM(IDC_CHAT_SEND, BN_CLICKED), (LPARAM)g_chat_send);
            return 0;
        }
        if (id == IDC_TERM_INPUT) {
            PostMessageW(parent, WM_COMMAND, MAKEWPARAM(IDC_TERM_RUN, BN_CLICKED), (LPARAM)g_term_run);
            return 0;
        }
    }

    if (hwnd == g_chat_input && g_chat_input_oldproc)
        return CallWindowProcW(g_chat_input_oldproc, hwnd, msg, wParam, lParam);
    if (hwnd == g_term_input && g_term_input_oldproc)
        return CallWindowProcW(g_term_input_oldproc, hwnd, msg, wParam, lParam);
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static void apply_theme(HWND hwnd)
{
    int i;
    recreate_brushes();
    recreate_fonts();
    set_title_dark(hwnd, theme_is_dark(g_theme_idx));

    /* apply mono font to code controls */
    if (g_font_mono) {
        if (g_editor)     SendMessageW(g_editor,     WM_SETFONT, (WPARAM)g_font_mono, FALSE);
        if (g_chat_log)   SendMessageW(g_chat_log,   WM_SETFONT, (WPARAM)g_font_mono, FALSE);
        if (g_term_log)   SendMessageW(g_term_log,   WM_SETFONT, (WPARAM)g_font_mono, FALSE);
        if (g_outline)    SendMessageW(g_outline,    WM_SETFONT, (WPARAM)g_font_mono, FALSE);
    }
    /* apply ui font to input + status */
    if (g_font_ui) {
        if (g_chat_input)   SendMessageW(g_chat_input,   WM_SETFONT, (WPARAM)g_font_ui, FALSE);
        if (g_term_input)   SendMessageW(g_term_input,   WM_SETFONT, (WPARAM)g_font_ui, FALSE);
        if (g_status)       SendMessageW(g_status,       WM_SETFONT, (WPARAM)g_font_ui, FALSE);
        if (g_app_subtitle) SendMessageW(g_app_subtitle, WM_SETFONT, (WPARAM)g_font_ui, FALSE);
        if (g_theme_combo)  SendMessageW(g_theme_combo,  WM_SETFONT, (WPARAM)g_font_ui, FALSE);
    }
    if (g_font_title && g_app_title) {
        SendMessageW(g_app_title, WM_SETFONT, (WPARAM)g_font_title, FALSE);
    }
    /* bold ui font for toolbar buttons */
    {
        HFONT bf = g_font_ui_bold ? g_font_ui_bold : g_font_ui;
        if (g_btn_add_chat)     { SendMessageW(g_btn_add_chat,    WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_btn_add_chat,    NULL, TRUE); }
        if (g_btn_add_editor)   { SendMessageW(g_btn_add_editor,  WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_btn_add_editor,  NULL, TRUE); }
        if (g_btn_add_terminal) { SendMessageW(g_btn_add_terminal,WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_btn_add_terminal, NULL, TRUE); }
        if (g_btn_add_outline)  { SendMessageW(g_btn_add_outline, WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_btn_add_outline,  NULL, TRUE); }
        if (g_chat_send)        { SendMessageW(g_chat_send,       WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_chat_send,        NULL, TRUE); }
        if (g_term_run)         { SendMessageW(g_term_run,        WM_SETFONT, (WPARAM)bf, FALSE); InvalidateRect(g_term_run,         NULL, TRUE); }
    }
    /* bold ui for header labels */
    if (g_font_ui_bold) {
        HWND *lbls[] = { &g_lbl_chat, &g_lbl_editor, &g_lbl_outline, &g_lbl_term };
        int k;
        for (k = 0; k < 4; k++) if (*lbls[k])
            SendMessageW(*lbls[k], WM_SETFONT, (WPARAM)g_font_ui_bold, FALSE);
    }

    /* repaint all */
    for (i = IDC_BOOT_TEXT; i <= IDC_APP_SUBTITLE; i++) {
        HWND h = GetDlgItem(hwnd, i);
        if (h) InvalidateRect(h, NULL, TRUE);
    }
    InvalidateRect(hwnd, NULL, TRUE);
}

static void layout_workspace(HWND hwnd)
{
        RECT rc;
        int  w, h;
        const int PAD      = 10;
        const int TOP_H    = 78;
        const int STATUS_H = 22;
        const int HDR_H    = 26;   /* section header strip height */
        const int INPUT_H  = 36;

        int left_x, left_y, left_w, left_h;
        int right_x, right_y, right_w, right_h;
        int editor_slot = 0, outline_slot = 0;
        int chat_slot   = 0, term_slot    = 0;
        int cur_y;

        if (!g_boot_done) return;

        GetClientRect(hwnd, &rc);
        w = rc.right  - rc.left;
        h = rc.bottom - rc.top;

        /* top bar */
        MoveWindow(g_app_title,        PAD,          PAD - 2, 420, 26, TRUE);
        MoveWindow(g_app_subtitle,     PAD,          PAD + 21, 560, 18, TRUE);

        MoveWindow(g_btn_add_chat,     PAD,          PAD + 42, 102, 28, TRUE);
        MoveWindow(g_btn_add_editor,   PAD + 108,    PAD + 42, 108, 28, TRUE);
        MoveWindow(g_btn_add_terminal, PAD + 222,    PAD + 42, 118, 28, TRUE);
        MoveWindow(g_btn_add_outline,  PAD + 346,    PAD + 42, 112, 28, TRUE);
        MoveWindow(g_theme_combo,      w - 246,      PAD + 42, 236, 280, TRUE);

        /* content area bounds */
        left_x  = PAD;
        left_y  = PAD + TOP_H;
        left_w  = (w - PAD*3) / 2;
        left_h  = h - left_y - PAD - STATUS_H;
        right_x = left_x + left_w + PAD;
        right_y = left_y;
        right_w = w - right_x - PAD;
        right_h = left_h;

        /* ── left column: editor + outline ──── */
        {
            int available = left_h;
            int n = (g_show_editor ? 1 : 0) + (g_show_outline ? 1 : 0);
            if (n >= 2) {
                /* account for the two headers and a gap */
                int body = available - HDR_H*2 - PAD;
                editor_slot  = (body * 70) / 100;
                outline_slot = body - editor_slot;
            } else if (n == 1) {
                editor_slot  = g_show_editor  ? (available - HDR_H) : 0;
                outline_slot = g_show_outline ? (available - HDR_H) : 0;
            }
        }

        cur_y = left_y;
        if (g_show_editor) {
            ShowWindow(g_lbl_editor, SW_SHOW);
            MoveWindow(g_lbl_editor, left_x, cur_y, left_w, HDR_H, TRUE);
            cur_y += HDR_H;
            ShowWindow(g_editor, SW_SHOW);
            MoveWindow(g_editor, left_x, cur_y, left_w, editor_slot, TRUE);
            cur_y += editor_slot + PAD;
        } else {
            ShowWindow(g_lbl_editor, SW_HIDE);
            ShowWindow(g_editor,     SW_HIDE);
        }
        if (g_show_outline) {
            ShowWindow(g_lbl_outline, SW_SHOW);
            MoveWindow(g_lbl_outline, left_x, cur_y, left_w, HDR_H, TRUE);
            cur_y += HDR_H;
            ShowWindow(g_outline, SW_SHOW);
            MoveWindow(g_outline, left_x, cur_y, left_w, outline_slot, TRUE);
        } else {
            ShowWindow(g_lbl_outline, SW_HIDE);
            ShowWindow(g_outline,     SW_HIDE);
        }

        /* ── right column: chat + terminal ──── */
        {
            int available = right_h;
            int n = (g_show_chat ? 1 : 0) + (g_show_terminal ? 1 : 0);
            if (n >= 2) {
                int body = available - HDR_H*2 - PAD - INPUT_H*2 - PAD*2;
                chat_slot = (body * 62) / 100;
                term_slot = body - chat_slot;
            } else if (g_show_chat) {
                chat_slot = available - HDR_H - INPUT_H - PAD;
                term_slot = 0;
            } else if (g_show_terminal) {
                chat_slot = 0;
                term_slot = available - HDR_H - INPUT_H - PAD;
            }
        }

        cur_y = right_y;
        if (g_show_chat) {
            ShowWindow(g_lbl_chat,   SW_SHOW);
            MoveWindow(g_lbl_chat,   right_x, cur_y, right_w, HDR_H, TRUE);
            cur_y += HDR_H;
            ShowWindow(g_chat_log,   SW_SHOW);
            MoveWindow(g_chat_log,   right_x, cur_y, right_w, chat_slot, TRUE);
            cur_y += chat_slot + PAD;
            ShowWindow(g_chat_input, SW_SHOW);
            ShowWindow(g_chat_send,  SW_SHOW);
            MoveWindow(g_chat_input, right_x,            cur_y, right_w - 90, INPUT_H, TRUE);
            MoveWindow(g_chat_send,  right_x+right_w-84, cur_y, 84,           INPUT_H, TRUE);
            cur_y += INPUT_H + PAD;
        } else {
            ShowWindow(g_lbl_chat,   SW_HIDE);
            ShowWindow(g_chat_log,   SW_HIDE);
            ShowWindow(g_chat_input, SW_HIDE);
            ShowWindow(g_chat_send,  SW_HIDE);
        }
        if (g_show_terminal) {
            ShowWindow(g_lbl_term,   SW_SHOW);
            MoveWindow(g_lbl_term,   right_x, cur_y, right_w, HDR_H, TRUE);
            cur_y += HDR_H;
            ShowWindow(g_term_log,   SW_SHOW);
            MoveWindow(g_term_log,   right_x, cur_y, right_w, term_slot, TRUE);
            cur_y += term_slot + PAD;
            ShowWindow(g_term_input, SW_SHOW);
            ShowWindow(g_term_run,   SW_SHOW);
            MoveWindow(g_term_input, right_x,            cur_y, right_w - 74, INPUT_H, TRUE);
            MoveWindow(g_term_run,   right_x+right_w-68, cur_y, 68,           INPUT_H, TRUE);
        } else {
            ShowWindow(g_lbl_term,   SW_HIDE);
            ShowWindow(g_term_log,   SW_HIDE);
            ShowWindow(g_term_input, SW_HIDE);
            ShowWindow(g_term_run,   SW_HIDE);
        }

    MoveWindow(g_status, PAD, h-PAD-STATUS_H, w-PAD*2, STATUS_H, TRUE);
    InvalidateRect(hwnd, NULL, TRUE);
}

static void create_workspace_controls(HWND hwnd)
{
    int i;
    /* ensure fonts exist before assigning */
    if (!g_font_ui) recreate_fonts();

#define MKBTN(var,lbl,id)  (var) = CreateWindowExW(0, L"BUTTON", (lbl), \
        WS_CHILD|WS_VISIBLE|BS_OWNERDRAW, 0,0,10,10, hwnd,(HMENU)(id),NULL,NULL)

    /* ── toolbar buttons (owner-draw → flat look) ──────────────────────────── */
    MKBTN(g_btn_add_chat,    L"Chat On",      IDC_ADD_CHAT);
    MKBTN(g_btn_add_editor,  L"Editor On",    IDC_ADD_EDITOR);
    MKBTN(g_btn_add_terminal,L"Terminal Off", IDC_ADD_TERMINAL);
    MKBTN(g_btn_add_outline, L"Outline Off",  IDC_ADD_OUTLINE);
#undef MKBTN

    g_app_title = CreateWindowExW(0, L"STATIC", L"TensorChat Studio",
                                  WS_CHILD|WS_VISIBLE|SS_LEFT,
                                  0,0,10,10, hwnd,(HMENU)IDC_APP_TITLE,NULL,NULL);
    g_app_subtitle = CreateWindowExW(0, L"STATIC", L"Fast local workspace for chat, coding, and terminal loops",
                                     WS_CHILD|WS_VISIBLE|SS_LEFT,
                                     0,0,10,10, hwnd,(HMENU)IDC_APP_SUBTITLE,NULL,NULL);

    /* ── theme combo ────────────────────────────────────────────────────────── */
    g_theme_combo = CreateWindowExW(0, WC_COMBOBOXW, L"",
                                    WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST|WS_VSCROLL,
                                    0,0,10,240, hwnd,(HMENU)IDC_THEME_COMBO,NULL,NULL);
    strip_ctrl_theme(g_theme_combo);
    for (i = 0; i < THEME_COUNT; i++)
        SendMessageW(g_theme_combo, CB_ADDSTRING, 0, (LPARAM)g_themes[i].name);
    SendMessageW(g_theme_combo, CB_SETCURSEL, g_theme_idx, 0);

    /* ── status bar ─────────────────────────────────────────────────────────── */
    g_status = CreateWindowExW(0, L"STATIC",
                               L"Ready. Use module toggles in the top bar.",
                               WS_CHILD|WS_VISIBLE|SS_LEFT,
                               0,0,10,22, hwnd,(HMENU)IDC_STATUS,NULL,NULL);
    SendMessageW(g_status, WM_SETFONT, (WPARAM)g_font_ui, TRUE);

    /* ── section header labels ──────────────────────────────────────────────── */
    g_lbl_chat   = CreateWindowExW(0, L"STATIC", L"  CHAT",
                                   WS_CHILD|SS_LEFT|SS_CENTERIMAGE,
                                   0,0,10,28, hwnd,(HMENU)IDC_LBL_CHAT_HDR,NULL,NULL);
    g_lbl_editor = CreateWindowExW(0, L"STATIC", L"  EDITOR",
                                   WS_CHILD|SS_LEFT|SS_CENTERIMAGE,
                                   0,0,10,28, hwnd,(HMENU)IDC_LBL_EDIT_HDR,NULL,NULL);
    g_lbl_outline= CreateWindowExW(0, L"STATIC", L"  OUTLINE",
                                   WS_CHILD|SS_LEFT|SS_CENTERIMAGE,
                                   0,0,10,28, hwnd,(HMENU)IDC_LBL_OUTL_HDR,NULL,NULL);
    g_lbl_term   = CreateWindowExW(0, L"STATIC", L"  TERMINAL",
                                   WS_CHILD|SS_LEFT|SS_CENTERIMAGE,
                                   0,0,10,28, hwnd,(HMENU)IDC_LBL_TERM_HDR,NULL,NULL);

    /* ── editor panel ───────────────────────────────────────────────────────── */
    g_editor = CreateWindowExW(0, L"EDIT",
                               L"// Editor\r\nint main() {\r\n    return 0;\r\n}\r\n",
                               WS_CHILD|WS_BORDER|ES_LEFT|ES_MULTILINE|ES_AUTOVSCROLL|
                               ES_WANTRETURN|WS_VSCROLL,
                               0,0,10,10, hwnd,(HMENU)IDC_EDITOR,NULL,NULL);
    strip_ctrl_theme(g_editor);
    SendMessageW(g_editor, WM_SETFONT, (WPARAM)g_font_mono, TRUE);

    /* ── outline panel ──────────────────────────────────────────────────────── */
    g_outline = CreateWindowExW(0, L"LISTBOX", L"",
                                WS_CHILD|WS_BORDER|LBS_NOTIFY|WS_VSCROLL,
                                0,0,10,10, hwnd,(HMENU)IDC_OUTLINE,NULL,NULL);
    strip_ctrl_theme(g_outline);
    SendMessageW(g_outline, WM_SETFONT, (WPARAM)g_font_mono, TRUE);

    /* ── chat log ───────────────────────────────────────────────────────────── */
    g_chat_log = CreateWindowExW(0, L"EDIT",
                                 L"Welcome to TensorChat Studio.\r\n"
                                 L"Tips:\r\n"
                                 L"- Press Enter to send\r\n"
                                 L"- Use toggles to show or hide modules\r\n"
                                 L"- Type 'help' for built-in commands\r\n\r\n",
                                 WS_CHILD|WS_BORDER|ES_LEFT|ES_MULTILINE|
                                 ES_AUTOVSCROLL|ES_READONLY|WS_VSCROLL,
                                 0,0,10,10, hwnd,(HMENU)IDC_CHAT_LOG,NULL,NULL);
    strip_ctrl_theme(g_chat_log);
    SendMessageW(g_chat_log, WM_SETFONT, (WPARAM)g_font_mono, TRUE);

    /* ── chat input ─────────────────────────────────────────────────────────── */
    g_chat_input = CreateWindowExW(0, L"EDIT", L"",
                                   WS_CHILD|WS_BORDER|ES_LEFT|ES_AUTOHSCROLL,
                                   0,0,10,10, hwnd,(HMENU)IDC_CHAT_INPUT,NULL,NULL);
    strip_ctrl_theme(g_chat_input);
    g_chat_input_oldproc = (WNDPROC)SetWindowLongPtrW(g_chat_input, GWLP_WNDPROC, (LONG_PTR)input_edit_proc);
    SendMessageW(g_chat_input, WM_SETFONT, (WPARAM)g_font_ui, TRUE);

    /* ── chat send (owner-draw button) ──────────────────────────────────────── */
    g_chat_send = CreateWindowExW(0, L"BUTTON", L"Send",
                                  WS_CHILD|WS_VISIBLE|BS_OWNERDRAW,
                                  0,0,10,10, hwnd,(HMENU)IDC_CHAT_SEND,NULL,NULL);

    /* ── terminal log ───────────────────────────────────────────────────────── */
    g_term_log = CreateWindowExW(0, L"EDIT",
                                 L"Terminal (sandbox).\r\nType 'help' for commands.\r\n\r\n",
                                 WS_CHILD|WS_BORDER|ES_LEFT|ES_MULTILINE|
                                 ES_AUTOVSCROLL|ES_READONLY|WS_VSCROLL,
                                 0,0,10,10, hwnd,(HMENU)IDC_TERM_LOG,NULL,NULL);
    strip_ctrl_theme(g_term_log);
    SendMessageW(g_term_log, WM_SETFONT, (WPARAM)g_font_mono, TRUE);

    /* ── terminal input ─────────────────────────────────────────────────────── */
    g_term_input = CreateWindowExW(0, L"EDIT", L"",
                                   WS_CHILD|WS_BORDER|ES_LEFT|ES_AUTOHSCROLL,
                                   0,0,10,10, hwnd,(HMENU)IDC_TERM_INPUT,NULL,NULL);
    strip_ctrl_theme(g_term_input);
    g_term_input_oldproc = (WNDPROC)SetWindowLongPtrW(g_term_input, GWLP_WNDPROC, (LONG_PTR)input_edit_proc);
    SendMessageW(g_term_input, WM_SETFONT, (WPARAM)g_font_ui, TRUE);

    /* ── terminal run (owner-draw) ──────────────────────────────────────────── */
    g_term_run = CreateWindowExW(0, L"BUTTON", L"Run",
                                 WS_CHILD|WS_VISIBLE|BS_OWNERDRAW,
                                 0,0,10,10, hwnd,(HMENU)IDC_TERM_RUN,NULL,NULL);

    update_module_button_labels();
    update_outline();
    layout_workspace(hwnd);
    (void)i;
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_CREATE: {
        HFONT title_font = CreateFontW(28, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                       DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                       CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");

        g_boot_text = CreateWindowExW(0, L"STATIC",
                                      L"TensorChat Native is warming up...",
                                      WS_CHILD | WS_VISIBLE | SS_CENTER,
                                      0, 0, 200, 34, hwnd, (HMENU)IDC_BOOT_TEXT, NULL, NULL);
        g_boot_progress = CreateWindowExW(0, PROGRESS_CLASSW, L"",
                                          WS_CHILD | WS_VISIBLE | PBS_MARQUEE,
                                          0, 0, 200, 24, hwnd, (HMENU)IDC_BOOT_PROGRESS, NULL, NULL);
        SendMessageW(g_boot_text, WM_SETFONT, (WPARAM)title_font, TRUE);
        SendMessageW(g_boot_progress, PBM_SETMARQUEE, TRUE, 30);

        SetTimer(hwnd, BOOT_TIMER_ID, 1400, NULL);
        recreate_brushes();
        g_main_hwnd = hwnd;
        return 0;
    }

    case WM_TIMER:
        if (wParam == BOOT_TIMER_ID && !g_boot_done) {
            KillTimer(hwnd, BOOT_TIMER_ID);
            ShowWindow(g_boot_text, SW_HIDE);
            ShowWindow(g_boot_progress, SW_HIDE);
            g_boot_done = 1;
            create_workspace_controls(hwnd);
            apply_theme(hwnd);
            set_status(L"Ready. Toggle modules from the top bar.");
        }
        return 0;

    case WM_SIZE:
        if (!g_boot_done) {
            RECT rc;
            int w, h;
            GetClientRect(hwnd, &rc);
            w = rc.right - rc.left;
            h = rc.bottom - rc.top;
            MoveWindow(g_boot_text, w / 2 - 260, h / 2 - 48, 520, 34, TRUE);
            MoveWindow(g_boot_progress, w / 2 - 220, h / 2 + 4, 440, 22, TRUE);
        } else {
            layout_workspace(hwnd);
        }
        return 0;

    case WM_COMMAND: {
        int id = LOWORD(wParam);
        int code = HIWORD(wParam);

        if (id == IDC_CHAT_SEND && code == BN_CLICKED) {
            on_chat_send();
            return 0;
        }
        if (id == IDC_TERM_RUN && code == BN_CLICKED) {
            on_term_run();
            return 0;
        }
        if (id == IDC_ADD_CHAT && code == BN_CLICKED) {
            if (g_show_chat && visible_module_count() == 1) {
                set_status(L"At least one module must stay visible.");
                return 0;
            }
            g_show_chat = !g_show_chat;
            update_module_button_labels();
            layout_workspace(hwnd);
            set_status(g_show_chat ? L"Chat enabled." : L"Chat hidden.");
            return 0;
        }
        if (id == IDC_ADD_EDITOR && code == BN_CLICKED) {
            if (g_show_editor && visible_module_count() == 1) {
                set_status(L"At least one module must stay visible.");
                return 0;
            }
            g_show_editor = !g_show_editor;
            update_module_button_labels();
            layout_workspace(hwnd);
            set_status(g_show_editor ? L"Editor enabled." : L"Editor hidden.");
            return 0;
        }
        if (id == IDC_ADD_TERMINAL && code == BN_CLICKED) {
            if (g_show_terminal && visible_module_count() == 1) {
                set_status(L"At least one module must stay visible.");
                return 0;
            }
            g_show_terminal = !g_show_terminal;
            update_module_button_labels();
            layout_workspace(hwnd);
            set_status(g_show_terminal ? L"Terminal enabled." : L"Terminal hidden.");
            return 0;
        }
        if (id == IDC_ADD_OUTLINE && code == BN_CLICKED) {
            if (g_show_outline && visible_module_count() == 1) {
                set_status(L"At least one module must stay visible.");
                return 0;
            }
            g_show_outline = !g_show_outline;
            update_module_button_labels();
            update_outline();
            layout_workspace(hwnd);
            set_status(g_show_outline ? L"Outline enabled." : L"Outline hidden.");
            return 0;
        }
        if (id == IDC_THEME_COMBO && code == CBN_SELCHANGE) {
            int sel = (int)SendMessageW(g_theme_combo, CB_GETCURSEL, 0, 0);
            if (sel >= 0 && sel < (int)(sizeof(g_themes) / sizeof(g_themes[0]))) {
                g_theme_idx = sel;
                apply_theme(hwnd);
                set_status(L"Theme switched.");
            }
            return 0;
        }
        if (id == IDC_EDITOR && code == EN_CHANGE) {
            if (g_show_outline) update_outline();
            return 0;
        }
        break;
    }

    case WM_APP_CHAT_REPLY: {
        chat_async_ctx_t *ctx = (chat_async_ctx_t *)lParam;
        if (ctx) {
            wchar_t bot_msg[2400];
            wchar_t t[16];
            current_time_tag(t, (int)(sizeof(t) / sizeof(t[0])));
            wsprintfW(bot_msg,
                      L"%s TensorChat: %s\r\n----------------------------------------\r\n\r\n",
                      t, ctx->reply);
            append_text(g_chat_log, bot_msg);
            g_chat_waiting = 0;
            EnableWindow(g_chat_send, TRUE);
            set_status(ctx->ok ? L"Runtime reply received."
                                : L"Reply produced with local fallback.");
            HeapFree(GetProcessHeap(), 0, ctx);
        }
        return 0;
    }

    case WM_APP_TERM_DONE: {
        term_async_ctx_t *ctx = (term_async_ctx_t *)lParam;
        if (ctx) {
            append_text(g_term_log, ctx->output);
            append_text(g_term_log, L"\r\n");
            HeapFree(GetProcessHeap(), 0, ctx);
        }
        return 0;
    }

    case WM_DRAWITEM: {
        LPDRAWITEMSTRUCT dis = (LPDRAWITEMSTRUCT)lParam;
        if (dis && dis->CtlType == ODT_BUTTON) {
            draw_button(dis);
            return TRUE;
        }
        break;
    }

    case WM_CTLCOLORSTATIC: {
        HDC  hdc = (HDC)wParam;
        HWND hctl= (HWND)lParam;
        SetBkMode(hdc, OPAQUE);
        if (hctl == g_lbl_chat || hctl == g_lbl_editor ||
            hctl == g_lbl_outline || hctl == g_lbl_term ||
            hctl == g_app_subtitle) {
            SetTextColor(hdc, g_themes[g_theme_idx].text_dim);
            SetBkColor(hdc,   g_themes[g_theme_idx].header);
            return (LRESULT)g_br_header;
        }
        if (hctl == g_app_title) {
            SetTextColor(hdc, g_themes[g_theme_idx].text);
            SetBkColor(hdc,   g_themes[g_theme_idx].bg);
            return (LRESULT)g_br_bg;
        }
        SetTextColor(hdc, g_themes[g_theme_idx].text);
        SetBkColor(hdc,   g_themes[g_theme_idx].panel);
        return (LRESULT)g_br_panel;
    }

    case WM_CTLCOLOREDIT:
    case WM_CTLCOLORLISTBOX: {
        HDC hdc = (HDC)wParam;
        SetBkMode(hdc, OPAQUE);
        SetTextColor(hdc, g_themes[g_theme_idx].text);
        SetBkColor(hdc,   g_themes[g_theme_idx].input);
        return (LRESULT)g_br_input;
    }

    case WM_CTLCOLORBTN: {
        HDC hdc = (HDC)wParam;
        SetBkMode(hdc, OPAQUE);
        SetTextColor(hdc, g_themes[g_theme_idx].btn_text);
        SetBkColor(hdc,   g_themes[g_theme_idx].btn_bg);
        return (LRESULT)g_br_btn;
    }

    case WM_ERASEBKGND: {
        RECT rc;
        HDC hdc = (HDC)wParam;
        GetClientRect(hwnd, &rc);
        FillRect(hdc, &rc, g_br_bg);
        return 1;
    }

    case WM_DESTROY:
        if (g_br_bg)     DeleteObject(g_br_bg);
        if (g_br_panel)  DeleteObject(g_br_panel);
        if (g_br_header) DeleteObject(g_br_header);
        if (g_br_input)  DeleteObject(g_br_input);
        if (g_br_btn)    DeleteObject(g_br_btn);
        if (g_font_ui)       DeleteObject(g_font_ui);
        if (g_font_ui_bold)  DeleteObject(g_font_ui_bold);
        if (g_font_mono)     DeleteObject(g_font_mono);
        if (g_font_title)    DeleteObject(g_font_title);
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEXW wc;
    MSG msg;
    HWND hwnd;
    INITCOMMONCONTROLSEX icc;

    (void)hPrevInstance;
    (void)lpCmdLine;

    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&icc);

    ZeroMemory(&wc, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = APP_CLASS_NAME;

    if (!RegisterClassExW(&wc)) return 1;

    hwnd = CreateWindowExW(0, APP_CLASS_NAME, L"TensorChat Native Studio",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           CW_USEDEFAULT, CW_USEDEFAULT, 1400, 860,
                           NULL, NULL, hInstance, NULL);
    if (!hwnd) return 1;

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}
