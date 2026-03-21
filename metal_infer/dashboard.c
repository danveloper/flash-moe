/*
 * Flash-MoE Dashboard — htop-style terminal monitor (ncurses)
 *
 * Reads /tmp/flash-moe-stats.json (written by the inference server)
 * and renders a live terminal dashboard every 500ms.
 *
 * Build:  make dashboard
 * Run:    ./dashboard [--port PORT]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include <ncurses.h>

// ---- Stats structure ----
typedef struct {
    char state[32];
    char request_id[64];
    int prefill_tokens;
    int prefill_done;
    int gen_tokens;
    int gen_max;
    double tok_per_sec;
    double elapsed_ms;
    double ttft_ms;
    int think_tokens;
    int total_requests;
    double uptime_s;
    char model[64];
    char quant[16];
    int k;
    int port;
    int connected;
} Stats;

// ---- Volatile flag for clean exit ----
static volatile int g_running = 1;

static void handle_sigint(int sig) {
    (void)sig;
    g_running = 0;
}

// ---- Simple JSON string extractor ----
static int json_get_str(const char *json, const char *key, char *dst, int dst_size) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) { dst[0] = '\0'; return 0; }
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') { dst[0] = '\0'; return 0; }
    p++;
    int i = 0;
    while (*p && *p != '"' && i < dst_size - 1) {
        dst[i++] = *p++;
    }
    dst[i] = '\0';
    return 1;
}

static double json_get_num(const char *json, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return atof(p);
}

// ---- Read stats from file ----
static void read_stats(Stats *s) {
    memset(s, 0, sizeof(*s));

    FILE *f = fopen("/tmp/flash-moe-stats.json", "r");
    if (!f) return;

    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return;
    buf[n] = '\0';

    struct stat st;
    if (stat("/tmp/flash-moe-stats.json", &st) == 0) {
        time_t now_t = time(NULL);
        if (now_t - st.st_mtime > 120) return;
    }

    s->connected = 1;
    json_get_str(buf, "state", s->state, sizeof(s->state));
    json_get_str(buf, "request_id", s->request_id, sizeof(s->request_id));
    json_get_str(buf, "model", s->model, sizeof(s->model));
    json_get_str(buf, "quant", s->quant, sizeof(s->quant));

    s->prefill_tokens = (int)json_get_num(buf, "prefill_tokens");
    s->prefill_done   = (int)json_get_num(buf, "prefill_done");
    s->gen_tokens     = (int)json_get_num(buf, "gen_tokens");
    s->gen_max        = (int)json_get_num(buf, "gen_max");
    s->tok_per_sec    = json_get_num(buf, "tok_per_sec");
    s->elapsed_ms     = json_get_num(buf, "elapsed_ms");
    s->ttft_ms        = json_get_num(buf, "ttft_ms");
    s->think_tokens   = (int)json_get_num(buf, "think_tokens");
    s->total_requests = (int)json_get_num(buf, "total_requests");
    s->uptime_s       = json_get_num(buf, "uptime_s");
    s->k              = (int)json_get_num(buf, "k");
    s->port           = (int)json_get_num(buf, "port");
}

// ---- Format uptime ----
static void format_uptime(double secs, char *buf, int buf_size) {
    int s = (int)secs;
    int h = s / 3600;
    int m = (s % 3600) / 60;
    if (h > 0)
        snprintf(buf, buf_size, "%dh %02dm", h, m);
    else if (m > 0)
        snprintf(buf, buf_size, "%dm %02ds", m, s % 60);
    else
        snprintf(buf, buf_size, "%ds", s);
}

// ---- Color pairs ----
#define CP_BORDER   1
#define CP_GREEN    2
#define CP_YELLOW   3
#define CP_RED      4
#define CP_MAGENTA  5
#define CP_GRAY     6
#define CP_BAR_FILL 7
#define CP_BAR_BG   8

// ---- Rolling averages ----
static double g_tok_history[120];
static int g_tok_count = 0;
static double g_ttft_history[1000];
static int g_ttft_count = 0;
static int g_last_requests = 0;

// ---- Draw a progress bar using ncurses ----
static void draw_bar(WINDOW *win, int y, int x, int width, double fraction,
                     int fill_color, int bg_color) {
    if (fraction < 0) fraction = 0;
    if (fraction > 1) fraction = 1;
    int filled = (int)(fraction * width + 0.5);
    if (filled > width) filled = width;

    wmove(win, y, x);
    wattron(win, COLOR_PAIR(fill_color));
    for (int i = 0; i < filled; i++)
        waddch(win, ACS_CKBOARD);
    wattroff(win, COLOR_PAIR(fill_color));

    wattron(win, COLOR_PAIR(bg_color));
    for (int i = filled; i < width; i++)
        waddch(win, ACS_CKBOARD);
    wattroff(win, COLOR_PAIR(bg_color));
}

// ---- Render ----
static void render(const Stats *s, int port_arg) {
    int term_h, term_w;
    getmaxyx(stdscr, term_h, term_w);
    (void)term_h;

    int box_w = term_w - 4;
    if (box_w < 40) box_w = 40;
    if (box_w > 120) box_w = 120;
    int box_x = 2;
    int inner_w = box_w - 2;  // content area inside borders

    char uptime_str[64];
    format_uptime(s->uptime_s, uptime_str, sizeof(uptime_str));

    // Track rolling averages
    if (s->connected && s->tok_per_sec > 0) {
        if (g_tok_count < 120) {
            g_tok_history[g_tok_count++] = s->tok_per_sec;
        } else {
            memmove(g_tok_history, g_tok_history + 1, 119 * sizeof(double));
            g_tok_history[119] = s->tok_per_sec;
        }
    }
    if (s->connected && s->ttft_ms > 0 && s->total_requests > g_last_requests) {
        if (g_ttft_count < 1000)
            g_ttft_history[g_ttft_count++] = s->ttft_ms;
        g_last_requests = s->total_requests;
    }

    double avg_tok = 0;
    for (int i = 0; i < g_tok_count; i++) avg_tok += g_tok_history[i];
    if (g_tok_count > 0) avg_tok /= g_tok_count;

    double avg_ttft = 0;
    for (int i = 0; i < g_ttft_count; i++) avg_ttft += g_ttft_history[i];
    if (g_ttft_count > 0) avg_ttft /= g_ttft_count;

    erase();

    // ---- Section 1: Title ----
    int row = 0;

    // Top border
    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x, ACS_ULCORNER);
    for (int i = 0; i < inner_w; i++) addch(ACS_HLINE);
    addch(ACS_URCORNER);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    row++;

    // Title line
    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x, ACS_VLINE);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x + inner_w + 1, ' ');  // clear for right border

    attron(A_BOLD);
    mvprintw(row, box_x + 2, "Flash-MoE Dashboard");
    attroff(A_BOLD);

    if (s->connected && s->tok_per_sec > 0) {
        attron(COLOR_PAIR(CP_GREEN) | A_BOLD);
        mvprintw(row, box_x + 24, "%5.1f tok/s", s->tok_per_sec);
        attroff(COLOR_PAIR(CP_GREEN) | A_BOLD);
        attron(COLOR_PAIR(CP_GRAY));
        printw("    up %s", uptime_str);
        attroff(COLOR_PAIR(CP_GRAY));
    } else if (s->connected) {
        attron(COLOR_PAIR(CP_GRAY));
        mvprintw(row, box_x + 44, "up %s", uptime_str);
        attroff(COLOR_PAIR(CP_GRAY));
    }

    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x + inner_w + 1, ACS_VLINE);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    row++;

    // Model info line
    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x, ACS_VLINE);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);

    if (s->connected) {
        attron(COLOR_PAIR(CP_GRAY));
        mvprintw(row, box_x + 2, "%s  %s  K=%d  Port %d",
                 s->model, s->quant, s->k, s->port > 0 ? s->port : port_arg);
        attroff(COLOR_PAIR(CP_GRAY));
    } else {
        attron(COLOR_PAIR(CP_GRAY));
        mvprintw(row, box_x + 2, "Waiting for server on port %d...", port_arg);
        attroff(COLOR_PAIR(CP_GRAY));
    }

    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x + inner_w + 1, ACS_VLINE);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    row++;

    // Divider
    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x, ACS_LTEE);
    for (int i = 0; i < inner_w; i++) addch(ACS_HLINE);
    addch(ACS_RTEE);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    row++;

    if (!s->connected) {
        // Disconnected state
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
        mvaddch(row, box_x, ACS_VLINE);
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
        attron(COLOR_PAIR(CP_RED) | A_BOLD);
        mvprintw(row, box_x + 2, "DISCONNECTED");
        attroff(COLOR_PAIR(CP_RED) | A_BOLD);
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
        mvaddch(row, box_x + inner_w + 1, ACS_VLINE);
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
        row++;

        attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
        mvaddch(row, box_x, ACS_VLINE);
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
        attron(COLOR_PAIR(CP_RED));
        mvprintw(row, box_x + 2, "Server not responding.");
        attroff(COLOR_PAIR(CP_RED));
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
        mvaddch(row, box_x + inner_w + 1, ACS_VLINE);
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
        row++;

        // Bottom border
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
        mvaddch(row, box_x, ACS_LLCORNER);
        for (int i = 0; i < inner_w; i++) addch(ACS_HLINE);
        addch(ACS_LRCORNER);
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
        row++;

        attron(COLOR_PAIR(CP_GRAY));
        mvprintw(row + 1, box_x, "Press Ctrl+C to exit");
        attroff(COLOR_PAIR(CP_GRAY));

        refresh();
        return;
    }

    int is_idle = (strcmp(s->state, "idle") == 0);
    int is_prefilling = (strcmp(s->state, "prefilling") == 0);
    int is_generating = (strcmp(s->state, "generating") == 0);

    // ---- Helper macro for bordered lines ----
    #define BORDER_LEFT() do { \
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD); \
        mvaddch(row, box_x, ACS_VLINE); \
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD); \
    } while(0)

    #define BORDER_RIGHT() do { \
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD); \
        mvaddch(row, box_x + inner_w + 1, ACS_VLINE); \
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD); \
    } while(0)

    #define DIVIDER() do { \
        attron(COLOR_PAIR(CP_BORDER) | A_BOLD); \
        mvaddch(row, box_x, ACS_LTEE); \
        for (int _i = 0; _i < inner_w; _i++) addch(ACS_HLINE); \
        addch(ACS_RTEE); \
        attroff(COLOR_PAIR(CP_BORDER) | A_BOLD); \
        row++; \
    } while(0)

    // ---- Section 2: Status ----

    // Status line
    BORDER_LEFT();
    if (is_generating) {
        mvprintw(row, box_x + 2, "Status: ");
        attron(COLOR_PAIR(CP_GREEN) | A_BOLD);
        printw("GENERATING");
        attroff(COLOR_PAIR(CP_GREEN) | A_BOLD);
        printw(" ");
        double gen_frac = s->gen_max > 0 ? (double)s->gen_tokens / s->gen_max : 0;
        draw_bar(stdscr, row, box_x + 22, 10, gen_frac, CP_GREEN, CP_GRAY);
        printw(" %d/%d tokens", s->gen_tokens, s->gen_max);
    } else if (is_prefilling) {
        mvprintw(row, box_x + 2, "Status: ");
        attron(COLOR_PAIR(CP_YELLOW) | A_BOLD);
        printw("PREFILLING");
        attroff(COLOR_PAIR(CP_YELLOW) | A_BOLD);
        printw(" ");
        double pf_frac = s->prefill_tokens > 0 ? (double)s->prefill_done / s->prefill_tokens : 0;
        draw_bar(stdscr, row, box_x + 22, 10, pf_frac, CP_YELLOW, CP_GRAY);
        printw(" %d/%d tokens", s->prefill_done, s->prefill_tokens);
    } else {
        mvprintw(row, box_x + 2, "Status: ");
        attron(A_DIM);
        printw("IDLE");
        attroff(A_DIM);
    }
    BORDER_RIGHT();
    row++;

    // Request line
    BORDER_LEFT();
    if (!is_idle && s->request_id[0]) {
        mvprintw(row, box_x + 2, "Request: %s   Elapsed: %.1fs",
                 s->request_id, s->elapsed_ms / 1000.0);
    } else if (s->request_id[0]) {
        attron(COLOR_PAIR(CP_GRAY));
        mvprintw(row, box_x + 2, "Last: %s", s->request_id);
        attroff(COLOR_PAIR(CP_GRAY));
    }
    BORDER_RIGHT();
    row++;

    // TTFT + Think line
    BORDER_LEFT();
    if (!is_idle) {
        if (s->ttft_ms > 0 && s->think_tokens > 0) {
            mvprintw(row, box_x + 2, "TTFT: %.1fs   Think: ", s->ttft_ms / 1000.0);
            attron(COLOR_PAIR(CP_MAGENTA));
            printw("%d tokens", s->think_tokens);
            attroff(COLOR_PAIR(CP_MAGENTA));
        } else if (s->ttft_ms > 0) {
            mvprintw(row, box_x + 2, "TTFT: %.1fs", s->ttft_ms / 1000.0);
        } else {
            attron(COLOR_PAIR(CP_GRAY));
            mvprintw(row, box_x + 2, "TTFT: --");
            attroff(COLOR_PAIR(CP_GRAY));
        }
    }
    BORDER_RIGHT();
    row++;

    // ---- Section 3: Progress bars ----
    DIVIDER();

    // Prefill bar
    {
        double pf_frac = s->prefill_tokens > 0 ? (double)s->prefill_done / s->prefill_tokens : 0;
        char suffix[64];
        snprintf(suffix, sizeof(suffix), " %3.0f%% (%d tokens)", pf_frac * 100, s->prefill_tokens);
        int label_w = 10;  // "Prefill:  "
        int suffix_w = (int)strlen(suffix);
        int bar_w = inner_w - label_w - suffix_w;
        if (bar_w < 5) bar_w = 5;

        BORDER_LEFT();
        int bar_color = (pf_frac >= 1.0) ? CP_GREEN : (is_prefilling ? CP_YELLOW : CP_GRAY);
        attron(COLOR_PAIR(CP_GREEN));
        mvprintw(row, box_x + 2, "Prefill:");
        attroff(COLOR_PAIR(CP_GREEN));
        printw("  ");
        draw_bar(stdscr, row, box_x + 2 + label_w, bar_w, pf_frac, bar_color, CP_BAR_BG);
        attron(COLOR_PAIR(CP_GREEN));
        mvprintw(row, box_x + 2 + label_w + bar_w, "%s", suffix);
        attroff(COLOR_PAIR(CP_GREEN));
        BORDER_RIGHT();
        row++;
    }

    // Generate bar
    {
        double gen_frac = s->gen_max > 0 ? (double)s->gen_tokens / s->gen_max : 0;
        char suffix[64];
        snprintf(suffix, sizeof(suffix), " %3.0f%% (%d/%d)", gen_frac * 100, s->gen_tokens, s->gen_max);
        int label_w = 10;  // "Generate: "
        int suffix_w = (int)strlen(suffix);
        int bar_w = inner_w - label_w - suffix_w;
        if (bar_w < 5) bar_w = 5;

        BORDER_LEFT();
        attron(COLOR_PAIR(CP_GREEN));
        mvprintw(row, box_x + 2, "Generate:");
        attroff(COLOR_PAIR(CP_GREEN));
        printw(" ");
        draw_bar(stdscr, row, box_x + 2 + label_w, bar_w, gen_frac,
                 is_generating ? CP_GREEN : CP_GRAY, CP_BAR_BG);
        attron(COLOR_PAIR(CP_GREEN));
        mvprintw(row, box_x + 2 + label_w + bar_w, "%s", suffix);
        attroff(COLOR_PAIR(CP_GREEN));
        BORDER_RIGHT();
        row++;
    }

    // ---- Section 4: Lifetime stats ----
    DIVIDER();

    // Row 1
    BORDER_LEFT();
    attron(COLOR_PAIR(CP_GREEN));
    mvprintw(row, box_x + 2, "Lifetime: ");
    attroff(COLOR_PAIR(CP_GREEN));
    printw("%d requests", s->total_requests);
    attron(COLOR_PAIR(CP_GRAY));
    printw("  |  ");
    attroff(COLOR_PAIR(CP_GRAY));
    attron(COLOR_PAIR(CP_GREEN));
    printw("Avg TTFT: ");
    attroff(COLOR_PAIR(CP_GREEN));
    if (avg_ttft > 0)
        printw("%.1fs", avg_ttft / 1000.0);
    else {
        attron(COLOR_PAIR(CP_GRAY));
        printw("--");
        attroff(COLOR_PAIR(CP_GRAY));
    }
    BORDER_RIGHT();
    row++;

    // Row 2
    BORDER_LEFT();
    attron(COLOR_PAIR(CP_GREEN));
    mvprintw(row, box_x + 2, "Avg tok/s: ");
    attroff(COLOR_PAIR(CP_GREEN));
    if (avg_tok > 0)
        printw("%.1f", avg_tok);
    else {
        attron(COLOR_PAIR(CP_GRAY));
        printw("--");
        attroff(COLOR_PAIR(CP_GRAY));
    }
    attron(COLOR_PAIR(CP_GRAY));
    printw("       |  ");
    attroff(COLOR_PAIR(CP_GRAY));
    attron(COLOR_PAIR(CP_GREEN));
    printw("Uptime: ");
    attroff(COLOR_PAIR(CP_GREEN));
    printw("%s", uptime_str);
    BORDER_RIGHT();
    row++;

    // Bottom border
    attron(COLOR_PAIR(CP_BORDER) | A_BOLD);
    mvaddch(row, box_x, ACS_LLCORNER);
    for (int i = 0; i < inner_w; i++) addch(ACS_HLINE);
    addch(ACS_LRCORNER);
    attroff(COLOR_PAIR(CP_BORDER) | A_BOLD);
    row++;

    attron(COLOR_PAIR(CP_GRAY));
    mvprintw(row + 1, box_x, "Press Ctrl+C to exit");
    attroff(COLOR_PAIR(CP_GRAY));

    refresh();

    #undef BORDER_LEFT
    #undef BORDER_RIGHT
    #undef DIVIDER
}

// ---- Main ----
int main(int argc, char **argv) {
    int port = 6601;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--port PORT]\n", argv[0]);
            printf("  --port PORT   Server port (default: 6601)\n");
            printf("\nReads /tmp/flash-moe-stats.json written by infer --serve\n");
            return 0;
        }
    }

    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    // Init ncurses
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    timeout(0);  // non-blocking getch

    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(CP_BORDER,   COLOR_CYAN,    -1);
        init_pair(CP_GREEN,    COLOR_GREEN,   -1);
        init_pair(CP_YELLOW,   COLOR_YELLOW,  -1);
        init_pair(CP_RED,      COLOR_RED,     -1);
        init_pair(CP_MAGENTA,  COLOR_MAGENTA, -1);
        init_pair(CP_GRAY,     COLOR_WHITE,   -1);  // closest to gray
        init_pair(CP_BAR_FILL, COLOR_GREEN,   COLOR_GREEN);
        init_pair(CP_BAR_BG,   COLOR_WHITE,   -1);
    }

    Stats stats;

    while (g_running) {
        int ch = getch();
        if (ch == 'q' || ch == 'Q') break;

        read_stats(&stats);
        render(&stats, port);
        usleep(500 * 1000);
    }

    // Cleanup
    endwin();
    printf("Dashboard exited.\n");

    return 0;
}
