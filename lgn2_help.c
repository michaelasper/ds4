#include "lgn2_help.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *off;
    const char *cyan;
    const char *title;
    const char *yellow;
    const char *grey;
    const char *red;
    const char *white;
    const char *bright;
} help_colors;

static help_colors help_make_colors(FILE *fp) {
    bool color = isatty(fileno(fp));
    help_colors c = {0};
    if (!color) return c;
    c.off = "\x1b[0m";
    c.cyan = "\x1b[38;5;81m";
    c.title = "\x1b[1;38;5;250m";
    c.yellow = "\x1b[38;5;179m";
    c.grey = "\x1b[38;5;240m";
    c.red = "\x1b[38;5;203m";
    c.white = "\x1b[38;5;252m";
    c.bright = "\x1b[1;38;5;231m";
    return c;
}

static void title(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->title ? c->title : "", s, c->off ? c->off : "");
}

static void title_red(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->red ? c->red : "", s, c->off ? c->off : "");
}

static bool option_name_has_switch(const char *name) {
    bool word_start = true;
    while (*name) {
        if (word_start && (*name == '-' || *name == '/')) return true;
        word_start = (*name == ' ');
        name++;
    }
    return false;
}

static void print_colored_option_name(FILE *fp, const help_colors *c, const char *name) {
    bool has_switch = option_name_has_switch(name);
    bool word_start = true;
    while (*name) {
        const char *start = name;
        while (*name && *name != ' ') name++;
        bool is_option = !has_switch || *start == '-' || *start == '/' ||
                         (word_start && has_switch && *start != '[');
        const char *color = is_option ? c->cyan : c->bright;
        if (color) fputs(color, fp);
        fwrite(start, 1, (size_t)(name - start), fp);
        if (color && c->off) fputs(c->off, fp);
        if (*name == ' ') {
            fputc(*name++, fp);
            word_start = false;
        }
    }
}

static void opt(FILE *fp, const help_colors *c, const char *name, const char *desc) {
    if (c->cyan) {
        fputs("  ", fp);
        print_colored_option_name(fp, c, name);
        fprintf(fp, " %s|%s ", c->grey ? c->grey : "", c->grey ? c->off : "");
        fprintf(fp, "%s%s%s\n", c->white ? c->white : "", desc,
                c->white ? c->off : "");
        return;
    }

    const int col = 30;
    int n = (int)strlen(name);
    if (n > col) {
        fprintf(fp, "  %s\n      %s\n", name, desc);
    } else {
        fprintf(fp, "  %-30s %s\n", name, desc);
    }
}

static void para(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n",
            c->yellow ? c->yellow : "", s, c->yellow ? c->off : "");
}

static bool streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static bool topic_is(const char *topic, const char *name) {
    return topic && strcmp(topic, name) == 0;
}

static const char *tool_name(lgn2_help_tool tool) {
    switch (tool) {
    case LGN2_HELP_LGN2: return "lgn2";
    case LGN2_HELP_SERVER: return "lgn2-server";
    case LGN2_HELP_BENCH: return "lgn2-bench";
    case LGN2_HELP_EVAL: return "lgn2-eval";
    }
    return "lgn2";
}

bool lgn2_help_version_requested(int argc, char *const argv[]) {
    return argc == 2 && argv && argv[1] && strcmp(argv[1], "--version") == 0;
}

void lgn2_help_print_version(FILE *fp, lgn2_help_tool tool) {
    fprintf(fp, "LagoonNebula %s %s (revision %s)\n",
            tool_name(tool), LGN2_RELEASE_LABEL, LGN2_BUILD_REVISION);
}

static const char *tool_usage(lgn2_help_tool tool) {
    switch (tool) {
    case LGN2_HELP_LGN2:
        return "Usage: lgn2 [(-p PROMPT | --prompt-file FILE)] [options]";
    case LGN2_HELP_SERVER:
        return "Usage: lgn2-server [options]";
    case LGN2_HELP_BENCH:
        return "Usage: lgn2-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]";
    case LGN2_HELP_EVAL:
        return "Usage: lgn2-eval [options]";
    }
    return "Usage: lgn2 [options]";
}

static const char *tool_summary(lgn2_help_tool tool) {
    switch (tool) {
    case LGN2_HELP_LGN2:
        return "Chat with a local Laguna S2.1 GGUF, run one-shot prompts, or inspect a model.";
    case LGN2_HELP_SERVER:
        return "Serve one loaded Laguna S2.1 GGUF through OpenAI, Responses, Anthropic, and completion-compatible HTTP APIs.";
    case LGN2_HELP_BENCH:
        return "Measure prefill, decode, context growth, and KV-cache size across repeatable context frontiers.";
    case LGN2_HELP_EVAL:
        return "Run the built-in reasoning, math, science, and security evaluation harness with a live terminal UI.";
    }
    return "";
}

static void print_laguna_dflash_options(FILE *fp, const help_colors *c) {
    opt(fp, c, "--dflash FILE", "Laguna DFlash support GGUF for greedy speculative decoding.");
    opt(fp, c, "--dflash-draft N", "Maximum DFlash draft positions per adaptive verification batch, 1..15. Metal default: 3");
    opt(fp, c, "--dflash-p-min P", "Stop before proposals below probability P, 0..1. Default: 0.4; 0 keeps fixed verifier width");
}

static void print_model_runtime(FILE *fp, const help_colors *c,
                                lgn2_help_tool tool, bool full) {
    title(fp, c, "Model And Runtime");
    opt(fp, c, "-m, --model FILE", "GGUF model path. Default: lgn2.gguf");
    opt(fp, c, "--metal", "Use Apple Metal (the only supported inference backend).");
    if (tool != LGN2_HELP_BENCH) {
        opt(fp, c, "-c, --ctx N", "Allocated context tokens.");
    }
    if (tool == LGN2_HELP_SERVER) {
        opt(fp, c, "-n, --tokens N", "Default max output tokens when clients omit a limit.");
    }
    opt(fp, c, "-t, --threads N", "CPU helper threads for host-side/reference work.");
    if (full) {
        if (tool != LGN2_HELP_BENCH) print_laguna_dflash_options(fp, c);
        opt(fp, c, "--quality", "Prefer exact kernels where faster approximate paths exist.");
        opt(fp, c, "--warm-weights", "Touch mapped tensor pages at startup to reduce first-use stalls.");
    }
    if (!full && (tool == LGN2_HELP_LGN2 || tool == LGN2_HELP_SERVER)) {
        print_laguna_dflash_options(fp, c);
    }
    fputc('\n', fp);
}

static void print_sampling(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "Prompt And Sampling");
    opt(fp, c, "-n, --tokens N", "Maximum generated tokens.");
    opt(fp, c, "--temp F", "Sampling temperature. 0 is greedy/deterministic.");
    opt(fp, c, "--top-k N", "Sample only from the N highest-scoring tokens. 0 disables.");
    opt(fp, c, "--top-p F", "Nucleus sampling probability.");
    opt(fp, c, "--min-p F", "Keep tokens scoring at least F times the top token.");
    opt(fp, c, "--seed N", "Sampling seed for reproducible non-greedy runs.");
    para(fp, c, "Laguna defaults to temperature 0.7, top-k 20, top-p 0.95, and min-p 0.05. Explicit options always win.");
    opt(fp, c, "--think", "Use normal thinking mode.");
    opt(fp, c, "--think-max", "Use Think Max when context is large enough.");
    opt(fp, c, "--nothink", "Disable thinking and ask for direct replies.");
    if (full) {
        opt(fp, c, "-sys, --system TEXT", "System prompt. Empty string disables the default where supported.");
        opt(fp, c, "-p, --prompt TEXT", "One-shot prompt text.");
        opt(fp, c, "--prompt-file FILE", "Read one-shot prompt text from FILE.");
        opt(fp, c, "--raw-prompt", "Tokenize the one-shot prompt without chat markers.");
    }
    fputc('\n', fp);
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c);

static void print_cli_specific(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "CLI Modes");
    opt(fp, c, "lgn2", "Start the interactive prompt.");
    opt(fp, c, "lgn2 -p TEXT", "Run one prompt and exit.");
    opt(fp, c, "lgn2 --prompt-file FILE", "Run a long prompt from a file and exit.");
    fputc('\n', fp);
    if (full) {
        print_cli_diagnostics(fp, c);
    }
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c) {
    title(fp, c, "Diagnostics And Data Collection");
    opt(fp, c, "--inspect", "Load the model and print a summary only.");
    opt(fp, c, "--dump-tokens", "Tokenize the prompt exactly as written, then exit.");
    opt(fp, c, "--dump-logits FILE", "Write full next-token logits as JSON.");
    opt(fp, c, "--dump-logprobs FILE", "Write greedy continuation top-logprobs as JSON.");
    opt(fp, c, "--logprobs-top-k N", "Alternatives stored by --dump-logprobs. Default: 20");
    opt(fp, c, "--decode-consistency N", "Compare N-token decode logits with a fresh full prefill.");
    opt(fp, c, "--perplexity-file FILE", "Score raw text with teacher-forced NLL.");
    fputc('\n', fp);
}

static void print_cli_commands(FILE *fp, const help_colors *c) {
    title_red(fp, c, "Interactive Commands");
    opt(fp, c, "/help", "Show interactive commands.");
    opt(fp, c, "/think, /think-max, /nothink", "Switch thinking mode.");
    opt(fp, c, "/ctx N", "Restart the interactive session with a new context size.");
    opt(fp, c, "/read FILE", "Read FILE and submit it as the next user message.");
    opt(fp, c, "/quit, /exit", "Leave the prompt.");
    opt(fp, c, "Ctrl+C", "Stop current generation and return to lgn2>.");
    fputc('\n', fp);
}

static void print_server_api(FILE *fp, const help_colors *c) {
    title(fp, c, "HTTP API");
    opt(fp, c, "--host HOST", "Bind address. Default: 127.0.0.1");
    opt(fp, c, "--port N", "Bind port. Default: 8000");
    opt(fp, c, "--cors", "Add Access-Control-Allow-* headers for browser JS clients.");
    opt(fp, c, "--trace FILE", "Write prompts, cache decisions, output, and tool calls.");
    opt(fp, c, "--batched-session N", "Keep N resident sessions and batch decode-ready requests.");
    opt(fp, c, "--mixed-prefill-quantum N", "Prefill chunk while generations are active. Default: 128");
    para(fp, c, "Endpoints: /v1/chat/completions, /v1/responses, /v1/completions, and /v1/messages.");
    para(fp, c, "The server exposes the loaded Laguna S2.1 model through every compatible endpoint.");
    fputc('\n', fp);
}

static void print_server_thinking(FILE *fp, const help_colors *c) {
    title(fp, c, "Server Thinking Defaults");
    para(fp, c, "Chat requests default to high-effort thinking.");
    para(fp, c, "reasoning_effort=max or output_config.effort=max requests Think Max.");
    para(fp, c, "Think Max requires --ctx >= 393216; smaller contexts use high.");
    para(fp, c, "thinking={type:disabled}, think=false, or model=laguna-s-2.1-chat selects non-thinking mode.");
    para(fp, c, "In thinking mode, client sampling knobs are ignored like the official API.");
    fputc('\n', fp);
}

static void print_kv_cache(FILE *fp, const help_colors *c) {
    title(fp, c, "Disk KV Cache");
    opt(fp, c, "--kv-disk-dir DIR", "Enable disk KV checkpoints in DIR.");
    opt(fp, c, "--kv-disk-space-mb N", "Disk budget. Default when enabled: 4096");
    opt(fp, c, "--kv-cache-min-tokens N", "Do not save/load checkpoints shorter than N. Default: 512");
    opt(fp, c, "--kv-cache-cold-max-tokens N", "Save cold first prompts up to N tokens. 0 disables. Default: 30000");
    opt(fp, c, "--kv-cache-continued-interval-tokens N", "Save aligned continued frontiers. 0 disables. Default: 10000");
    opt(fp, c, "--kv-cache-boundary-trim-tokens N", "Trim tail tokens for cold boundary saves. Default: 32");
    opt(fp, c, "--kv-cache-boundary-align-tokens N", "Align cold boundary saves to this multiple. Default: 2048");
    opt(fp, c, "--kv-cache-reject-different-quant", "Reject checkpoints written with different routed-expert quantization.");
    opt(fp, c, "--disable-exact-dsml-tool-replay", "Disable exact sampled DSML tool replay map.");
    opt(fp, c, "--tool-memory-max-ids N", "Exact tool-call IDs kept in RAM. Default: 100000");
    fputc('\n', fp);
}

static void print_bench_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Benchmark Input");
    opt(fp, c, "--prompt-file FILE", "Raw benchmark text; token sequence is sliced at each frontier.");
    opt(fp, c, "--chat-prompt-file FILE", "Render FILE as one no-thinking chat user message.");
    opt(fp, c, "-sys, --system TEXT", "System prompt used only with --chat-prompt-file.");
    fputc('\n', fp);
    title(fp, c, "Benchmark Sweep");
    opt(fp, c, "--ctx-start N", "First measured frontier. Default: 2048");
    opt(fp, c, "--ctx-max N", "Last measured frontier. Default: 32768");
    opt(fp, c, "--ctx-alloc N", "Allocated context. Default: ctx-max + gen-tokens + 1");
    opt(fp, c, "--step-mul F", "Multiplicative step. Default: 1");
    opt(fp, c, "--step-incr N", "Linear step when --step-mul is 1. Default: 2048");
    opt(fp, c, "--gen-tokens N", "Greedy decode tokens per frontier. 0 for pure prefill. Default: 128");
    opt(fp, c, "--csv FILE", "Write CSV there instead of stdout.");
    opt(fp, c, "--dump-frontier-logits-dir DIR", "Write one full-logit JSON file per frontier.");
    opt(fp, c, "--dump-frontier-logits-f32-dir DIR", "Write a raw headerless native-endian 32-bit float array in vocab order per frontier.");
    fputc('\n', fp);
}

static void print_eval_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Evaluation");
    opt(fp, c, "-n, --tokens N", "Max generated tokens per question. Default: 16000");
    opt(fp, c, "--questions N", "Run only the first N embedded questions.");
    opt(fp, c, "--case-sequence LIST", "Run 1-based case numbers in this comma-separated order.");
    opt(fp, c, "--trace FILE", "Write questions, outputs, and grading decisions.");
    opt(fp, c, "--regrade-trace FILE", "Regrade a prior trace without loading the model.");
    opt(fp, c, "--soft-limit-reply-budget N", "Soft close thinking near the end of reply budget. Default: 1024");
    opt(fp, c, "--hard-limit-reply-budget N", "Force </think> with N tokens left. Default: 512");
    opt(fp, c, "--soft-limit-think-close-rank N", "Soft-close when </think> is in top N tokens. Default: 3");
    opt(fp, c, "--pause-ms N", "Pause after each result in the TTY UI. Default: 350");
    opt(fp, c, "--plain", "Disable split-screen ANSI UI.");
    opt(fp, c, "--self-test-extractors", "Run answer-extractor self-tests and exit.");
    fputc('\n', fp);
}

static bool tool_has_topic(lgn2_help_tool tool, const char *topic) {
    if (!topic) return true;
    if (streq(topic, "all")) return true;
    if (streq(topic, "runtime")) return true;
    if (streq(topic, "sampling"))
        return tool == LGN2_HELP_LGN2 || tool == LGN2_HELP_EVAL;
    switch (tool) {
    case LGN2_HELP_LGN2:
        return streq(topic, "diagnostics") || streq(topic, "commands");
    case LGN2_HELP_SERVER:
        return streq(topic, "api") || streq(topic, "kv-cache") || streq(topic, "thinking");
    case LGN2_HELP_BENCH:
        return streq(topic, "benchmark");
    case LGN2_HELP_EVAL:
        return streq(topic, "evaluation");
    }
    return false;
}

static void more_line(FILE *fp, const help_colors *c, const char *label, const char *topic) {
    static const char *colors[] = {
        "\x1b[38;5;81m", "\x1b[38;5;114m", "\x1b[38;5;179m",
        "\x1b[38;5;141m", "\x1b[38;5;147m"
    };
    static size_t idx;
    const char *on = c->cyan ? colors[idx++ % (sizeof(colors) / sizeof(colors[0]))] : "";
    if (streq(label, "Interactive commands:") && c->red) on = c->red;
    const char *off = c->off ? c->off : "";
    fprintf(fp, "    %s%-26s%s --help %s\n", on, label, off, topic);
}

static void print_more_info(FILE *fp, const help_colors *c, lgn2_help_tool tool) {
    title(fp, c, "More Info");
    more_line(fp, c, "Runtime full info:", "runtime");
    if (tool_has_topic(tool, "sampling"))
        more_line(fp, c, "Sampling full info:", "sampling");
    if (tool == LGN2_HELP_LGN2) {
        more_line(fp, c, "Interactive commands:", "commands");
        more_line(fp, c, "Diagnostics:", "diagnostics");
    } else if (tool == LGN2_HELP_SERVER) {
        more_line(fp, c, "HTTP API:", "api");
        more_line(fp, c, "Disk KV cache:", "kv-cache");
        more_line(fp, c, "Thinking behavior:", "thinking");
    } else if (tool == LGN2_HELP_BENCH) {
        more_line(fp, c, "Benchmark sweep:", "benchmark");
    } else if (tool == LGN2_HELP_EVAL) {
        more_line(fp, c, "Evaluation options:", "evaluation");
    }
    fputc('\n', fp);
}

static void print_examples(FILE *fp, const help_colors *c, lgn2_help_tool tool, const char *topic) {
    title(fp, c, "Examples");
    if (topic_is(topic, "runtime")) {
        if (tool == LGN2_HELP_SERVER) {
            opt(fp, c, "Metal API", "./lgn2-server -m lgn2.gguf --metal --ctx 100000");
            opt(fp, c, "batched API", "./lgn2-server --batched-session 2 --host 127.0.0.1 --port 8000");
        } else if (tool == LGN2_HELP_BENCH) {
            opt(fp, c, "bench", "./lgn2-bench --prompt-file long.txt --ctx-max 32768");
        } else if (tool == LGN2_HELP_EVAL) {
            opt(fp, c, "eval", "./lgn2-eval --questions 10 --ctx 100000");
        } else {
        opt(fp, c, "Metal", "./lgn2 -m lgn2.gguf --metal -c 100000");
        }
    } else if (tool == LGN2_HELP_SERVER || topic_is(topic, "api") || topic_is(topic, "kv-cache")) {
        opt(fp, c, "local API", "./lgn2-server --ctx 100000 --kv-disk-dir ~/.lgn2/server-kv --kv-disk-space-mb 8192");
        opt(fp, c, "curl", "curl http://127.0.0.1:8000/v1/models");
    } else if (tool == LGN2_HELP_BENCH || topic_is(topic, "benchmark")) {
        opt(fp, c, "csv", "./lgn2-bench --prompt-file long.txt --ctx-max 32768 --csv speed.csv");
        opt(fp, c, "prefill only", "./lgn2-bench --prompt-file long.txt --gen-tokens 0");
    } else if (tool == LGN2_HELP_EVAL || topic_is(topic, "evaluation")) {
        opt(fp, c, "first 10", "./lgn2-eval --questions 10 --trace eval.trace");
        opt(fp, c, "plain", "./lgn2-eval --plain --nothink --tokens 512");
    } else {
        opt(fp, c, "chat", "./lgn2");
        opt(fp, c, "one shot", "./lgn2 -p \"Explain mmap in C\"");
        opt(fp, c, "long prompt", "./lgn2 --think-max --prompt-file prompt.txt --ctx 393216");
    }
    fputc('\n', fp);
}

static void print_topic(FILE *fp, const help_colors *c, lgn2_help_tool tool, const char *topic) {
    if (streq(topic, "all")) {
        print_model_runtime(fp, c, tool, true);
        if (tool_has_topic(tool, "sampling")) print_sampling(fp, c, true);
        if (tool == LGN2_HELP_LGN2) {
            print_cli_specific(fp, c, true);
            print_cli_commands(fp, c);
        } else if (tool == LGN2_HELP_SERVER) {
            print_server_api(fp, c);
            print_server_thinking(fp, c);
            print_kv_cache(fp, c);
        } else if (tool == LGN2_HELP_BENCH) {
            print_bench_specific(fp, c);
        } else if (tool == LGN2_HELP_EVAL) {
            print_eval_specific(fp, c);
        }
        return;
    }

    if (streq(topic, "runtime")) print_model_runtime(fp, c, tool, true);
    else if (streq(topic, "sampling")) print_sampling(fp, c, true);
    else if (tool == LGN2_HELP_LGN2 && streq(topic, "diagnostics")) print_cli_diagnostics(fp, c);
    else if (tool == LGN2_HELP_LGN2 && streq(topic, "commands")) print_cli_commands(fp, c);
    else if (tool == LGN2_HELP_SERVER && streq(topic, "api")) print_server_api(fp, c);
    else if (tool == LGN2_HELP_SERVER && streq(topic, "kv-cache")) print_kv_cache(fp, c);
    else if (tool == LGN2_HELP_SERVER && streq(topic, "thinking")) print_server_thinking(fp, c);
    else if (tool == LGN2_HELP_BENCH && streq(topic, "benchmark")) print_bench_specific(fp, c);
    else if (tool == LGN2_HELP_EVAL && streq(topic, "evaluation")) print_eval_specific(fp, c);
}

static void print_default(FILE *fp, const help_colors *c, lgn2_help_tool tool) {
    print_model_runtime(fp, c, tool, false);

    if (tool == LGN2_HELP_LGN2) {
        print_cli_specific(fp, c, true);
        print_sampling(fp, c, false);
    } else if (tool == LGN2_HELP_SERVER) {
        print_server_api(fp, c);
        print_kv_cache(fp, c);
    } else if (tool == LGN2_HELP_BENCH) {
        print_bench_specific(fp, c);
    } else if (tool == LGN2_HELP_EVAL) {
        print_eval_specific(fp, c);
    }
}

static void print_identity(FILE *fp, const help_colors *c) {
    title(fp, c, "Identity");
    opt(fp, c, "--version", "Print LagoonNebula, this executable, the development release label, and the source revision.");
    fputc('\n', fp);
}

void lgn2_help_print(FILE *fp, lgn2_help_tool tool, const char *topic) {
    help_colors c = help_make_colors(fp);
    if (topic && !tool_has_topic(tool, topic)) {
        fprintf(fp, "%s: unknown help topic '%s'\n\n", tool_name(tool), topic);
        topic = NULL;
    }

    fprintf(fp, "%s%s%s\n", c.bright ? c.bright : "", tool_name(tool), c.off ? c.off : "");
    fprintf(fp, "%s\n\n", tool_summary(tool));
    fprintf(fp, "%s\n\n", tool_usage(tool));
    print_identity(fp, &c);

    if (topic) print_topic(fp, &c, tool, topic);
    else {
        print_default(fp, &c, tool);
        print_more_info(fp, &c, tool);
    }
    print_examples(fp, &c, tool, topic);
}
