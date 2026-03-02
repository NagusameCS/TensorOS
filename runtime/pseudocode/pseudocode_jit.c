/* =============================================================================
 * TensorOS - Pseudocode JIT Runtime Implementation
 *
 * The Pseudocode language (by NaguSamecs) serves as the default runtime for
 * AI tasks in TensorOS. This implementation provides:
 *
 * 1. A lexer/parser for Pseudocode syntax
 * 2. Tensor-aware type checking and shape inference
 * 3. Lowering to Tensor IR
 * 4. Tiered JIT compilation (interpret → basic JIT → optimized JIT)
 * 5. Built-in AI operations (train, infer, deploy)
 * 6. Native git integration (commit, push from code)
 *
 * Example Pseudocode for TensorOS:
 *
 *   model TransformerLM:
 *       layer embedding: vocab=50257, dim=768
 *       layer attention: heads=12, dim=768
 *       layer dense: in=768, out=50257
 *
 *   tensor data = load "dataset.jsonl"
 *   train TransformerLM on data for 10 epochs
 *   save TransformerLM to "model.safetensors"
 *   git commit "Trained transformer model"
 *   deploy TransformerLM on port 8080
 *
 * =============================================================================*/

#include "runtime/pseudocode/pseudocode_jit.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Global JIT State
 * =============================================================================*/

static bool jit_initialized = false;
__attribute__((unused))
static uint64_t jit_compilation_count = 0;

int pseudocode_jit_init(void)
{
    if (jit_initialized) return 0;

    /* Initialize the JIT code cache */
    /* Allocate executable memory region for JIT output */
    jit_initialized = true;
    kprintf_debug("[JIT] Pseudocode JIT compiler initialized\n");
    return 0;
}

/* =============================================================================
 * Runtime Creation/Destruction
 * =============================================================================*/

pseudo_runtime_t *pseudo_runtime_create(void)
{
    pseudo_runtime_t *rt = (pseudo_runtime_t *)kmalloc(sizeof(pseudo_runtime_t));
    if (!rt) return NULL;

    kmemset(rt, 0, sizeof(*rt));
    return rt;
}

void pseudo_runtime_destroy(pseudo_runtime_t *rt)
{
    if (!rt) return;
    kfree(rt);
}

/* =============================================================================
 * Lexer Implementation
 * Tokenizes Pseudocode source into a stream of tokens
 * =============================================================================*/

static bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool is_digit(char c) { return c >= '0' && c <= '9'; }
static bool is_alnum(char c) { return is_alpha(c) || is_digit(c); }
static bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

typedef struct {
    const char *keyword;
    token_type_t type;
} keyword_entry_t;

static const keyword_entry_t keywords[] = {
    {"model",     TOK_KW_MODEL},
    {"layer",     TOK_KW_LAYER},
    {"tensor",    TOK_KW_TENSOR},
    {"train",     TOK_KW_TRAIN},
    {"infer",     TOK_KW_INFER},
    {"load",      TOK_KW_LOAD},
    {"save",      TOK_KW_SAVE},
    {"pipeline",  TOK_KW_PIPELINE},
    {"if",        TOK_KW_IF},
    {"else",      TOK_KW_ELSE},
    {"for",       TOK_KW_FOR},
    {"while",     TOK_KW_WHILE},
    {"return",    TOK_KW_RETURN},
    {"function",  TOK_KW_FUNCTION},
    {"import",    TOK_KW_IMPORT},
    {"from",      TOK_KW_FROM},
    {"as",        TOK_KW_AS},
    {"print",     TOK_KW_PRINT},
    {"git",       TOK_KW_GIT},
    {"deploy",    TOK_KW_DEPLOY},
    {"monitor",   TOK_KW_MONITOR},
    {"matmul",    TOK_KW_MATMUL},
    {"conv",      TOK_KW_CONV},
    {"attention",  TOK_KW_ATTENTION},
    {"softmax",   TOK_KW_SOFTMAX},
    {"relu",      TOK_KW_RELU},
    {"sigmoid",   TOK_KW_SIGMOID},
    {"layernorm", TOK_KW_LAYERNORM},
    {"dropout",   TOK_KW_DROPOUT},
    {"reshape",   TOK_KW_RESHAPE},
    {"transpose", TOK_KW_TRANSPOSE},
    {"true",      TOK_BOOL_LIT},
    {"false",     TOK_BOOL_LIT},
    {NULL, TOK_ERROR},
};

static token_type_t lookup_keyword(const char *start, uint32_t length)
{
    for (int i = 0; keywords[i].keyword; i++) {
        const char *kw = keywords[i].keyword;
        uint32_t kw_len = 0;
        while (kw[kw_len]) kw_len++;

        if (kw_len == length) {
            bool match = true;
            for (uint32_t j = 0; j < length; j++) {
                if (start[j] != kw[j]) { match = false; break; }
            }
            if (match) return keywords[i].type;
        }
    }
    return TOK_IDENT;
}

int pseudo_lex(const char *source, token_t *tokens, uint32_t max_tokens,
                uint32_t *token_count)
{
    const char *p = source;
    uint32_t count = 0;
    uint32_t line = 1, col = 1;

    while (*p && count < max_tokens) {
        /* Skip whitespace (not newlines) */
        while (is_space(*p)) { p++; col++; }

        if (!*p) break;

        token_t *tok = &tokens[count];
        tok->line = line;
        tok->column = col;
        tok->start = p;

        /* Newline */
        if (*p == '\n') {
            tok->type = TOK_NEWLINE;
            tok->length = 1;
            p++; line++; col = 1;
            count++;
            continue;
        }

        /* Comments */
        if (*p == '#') {
            while (*p && *p != '\n') p++;
            continue;
        }

        /* String literal */
        if (*p == '"') {
            p++; col++;
            tok->start = p;
            while (*p && *p != '"' && *p != '\n') { p++; col++; }
            tok->type = TOK_STRING_LIT;
            tok->length = p - tok->start;
            if (*p == '"') { p++; col++; }
            count++;
            continue;
        }

        /* Number */
        if (is_digit(*p) || (*p == '-' && is_digit(p[1]))) {
            const char *start = p;
            if (*p == '-') { p++; col++; }
            while (is_digit(*p)) { p++; col++; }
            if (*p == '.') {
                p++; col++;
                while (is_digit(*p)) { p++; col++; }
                tok->type = TOK_FLOAT_LIT;
                /* Parse float value */
                tok->value.float_val = 0; /* TODO: proper float parsing */
            } else {
                tok->type = TOK_INT_LIT;
                /* Parse int value */
                tok->value.int_val = 0;
                const char *q = start;
                bool neg = false;
                if (*q == '-') { neg = true; q++; }
                while (is_digit(*q)) {
                    tok->value.int_val = tok->value.int_val * 10 + (*q - '0');
                    q++;
                }
                if (neg) tok->value.int_val = -tok->value.int_val;
            }
            tok->length = p - start;
            count++;
            continue;
        }

        /* Identifier or keyword */
        if (is_alpha(*p)) {
            const char *start = p;
            while (is_alnum(*p)) { p++; col++; }
            tok->length = p - start;
            tok->type = lookup_keyword(start, tok->length);
            if (tok->type == TOK_BOOL_LIT) {
                tok->value.bool_val = (start[0] == 't');
            }
            count++;
            continue;
        }

        /* Operators and delimiters */
        tok->length = 1;
        switch (*p) {
        case '+': tok->type = TOK_PLUS; break;
        case '-':
            if (p[1] == '>') { tok->type = TOK_ARROW; tok->length = 2; p++; col++; }
            else tok->type = TOK_MINUS;
            break;
        case '*': tok->type = TOK_STAR; break;
        case '/': tok->type = TOK_SLASH; break;
        case '@': tok->type = TOK_AT; break;
        case '%': tok->type = TOK_PERCENT; break;
        case '=':
            if (p[1] == '=') { tok->type = TOK_EQUAL_EQUAL; tok->length = 2; p++; col++; }
            else if (p[1] == '>') { tok->type = TOK_FAT_ARROW; tok->length = 2; p++; col++; }
            else tok->type = TOK_EQUAL;
            break;
        case '!':
            if (p[1] == '=') { tok->type = TOK_NOT_EQUAL; tok->length = 2; p++; col++; }
            else { tok->type = TOK_ERROR; }
            break;
        case '<':
            if (p[1] == '=') { tok->type = TOK_LESS_EQUAL; tok->length = 2; p++; col++; }
            else tok->type = TOK_LESS;
            break;
        case '>':
            if (p[1] == '=') { tok->type = TOK_GREATER_EQUAL; tok->length = 2; p++; col++; }
            else tok->type = TOK_GREATER;
            break;
        case '(': tok->type = TOK_LPAREN; break;
        case ')': tok->type = TOK_RPAREN; break;
        case '[': tok->type = TOK_LBRACKET; break;
        case ']': tok->type = TOK_RBRACKET; break;
        case '{': tok->type = TOK_LBRACE; break;
        case '}': tok->type = TOK_RBRACE; break;
        case ',': tok->type = TOK_COMMA; break;
        case ':': tok->type = TOK_COLON; break;
        case ';': tok->type = TOK_SEMICOLON; break;
        case '.': tok->type = TOK_DOT; break;
        default:  tok->type = TOK_ERROR; break;
        }
        p++; col++;
        count++;
    }

    /* EOF token */
    if (count < max_tokens) {
        tokens[count].type = TOK_EOF;
        tokens[count].line = line;
        tokens[count].column = col;
        tokens[count].length = 0;
        count++;
    }

    if (token_count) *token_count = count;
    return 0;
}

/* =============================================================================
 * Parser (Recursive Descent)
 * =============================================================================*/

typedef struct {
    const token_t *tokens;
    uint32_t       count;
    uint32_t       pos;
} parser_state_t;

static ast_node_t *alloc_node(ast_node_type_t type)
{
    ast_node_t *node = (ast_node_t *)kmalloc(sizeof(ast_node_t));
    if (node) {
        kmemset(node, 0, sizeof(*node));
        node->type = type;
    }
    return node;
}

static token_t *peek(parser_state_t *ps)
{
    if (ps->pos >= ps->count) return NULL;
    return (token_t *)&ps->tokens[ps->pos];
}

static token_t *advance(parser_state_t *ps)
{
    if (ps->pos >= ps->count) return NULL;
    return (token_t *)&ps->tokens[ps->pos++];
}

static void skip_newlines(parser_state_t *ps)
{
    while (ps->pos < ps->count && ps->tokens[ps->pos].type == TOK_NEWLINE)
        ps->pos++;
}

static ast_node_t *parse_expression(parser_state_t *ps);
static ast_node_t *parse_statement(parser_state_t *ps);

static ast_node_t *parse_primary(parser_state_t *ps)
{
    token_t *tok = peek(ps);
    if (!tok) return NULL;

    switch (tok->type) {
    case TOK_INT_LIT:
    case TOK_FLOAT_LIT:
    case TOK_STRING_LIT:
    case TOK_BOOL_LIT:
        {
            ast_node_t *node = alloc_node(AST_LITERAL);
            node->token = *advance(ps);
            return node;
        }

    case TOK_IDENT:
        {
            ast_node_t *node = alloc_node(AST_IDENT);
            node->token = *advance(ps);
            return node;
        }

    case TOK_LPAREN:
        {
            advance(ps); /* consume ( */
            ast_node_t *expr = parse_expression(ps);
            if (peek(ps) && peek(ps)->type == TOK_RPAREN)
                advance(ps); /* consume ) */
            return expr;
        }

    case TOK_LBRACKET:
        {
            /* Tensor shape expression: [dim1, dim2, ...] */
            ast_node_t *node = alloc_node(AST_SHAPE_EXPR);
            advance(ps); /* consume [ */
            /* Parse dimensions */
            while (peek(ps) && peek(ps)->type != TOK_RBRACKET) {
                ast_node_t *dim = parse_expression(ps);
                if (node->child_count < 8)
                    node->children[node->child_count++] = dim;
                if (peek(ps) && peek(ps)->type == TOK_COMMA)
                    advance(ps);
            }
            if (peek(ps) && peek(ps)->type == TOK_RBRACKET)
                advance(ps);
            return node;
        }

    default:
        return NULL;
    }
}

static ast_node_t *parse_expression(parser_state_t *ps)
{
    ast_node_t *left = parse_primary(ps);
    if (!left) return NULL;

    token_t *op = peek(ps);
    if (op && (op->type == TOK_PLUS || op->type == TOK_MINUS ||
               op->type == TOK_STAR || op->type == TOK_SLASH ||
               op->type == TOK_AT)) {
        ast_node_t *node = alloc_node(AST_BINARY_OP);
        node->token = *advance(ps);
        node->children[0] = left;
        node->children[1] = parse_expression(ps);
        node->child_count = 2;
        return node;
    }

    return left;
}

static ast_node_t *parse_statement(parser_state_t *ps)
{
    skip_newlines(ps);
    token_t *tok = peek(ps);
    if (!tok || tok->type == TOK_EOF) return NULL;

    switch (tok->type) {
    case TOK_KW_MODEL:
        {
            ast_node_t *node = alloc_node(AST_MODEL_DEF);
            node->token = *advance(ps); /* consume 'model' */
            /* Model name */
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            /* Skip colon */
            if (peek(ps) && peek(ps)->type == TOK_COLON)
                advance(ps);
            /* Parse layers */
            skip_newlines(ps);
            while (peek(ps) && peek(ps)->type == TOK_KW_LAYER) {
                ast_node_t *layer = alloc_node(AST_LAYER_DEF);
                layer->token = *advance(ps);
                /* Layer name and params */
                if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                    layer->children[0] = alloc_node(AST_IDENT);
                    layer->children[0]->token = *advance(ps);
                    layer->child_count = 1;
                }
                /* Skip to next line */
                while (peek(ps) && peek(ps)->type != TOK_NEWLINE && peek(ps)->type != TOK_EOF)
                    advance(ps);
                skip_newlines(ps);
                /* Add layer to model */
                if (node->child_count < 8)
                    node->children[node->child_count++] = layer;
            }
            return node;
        }

    case TOK_KW_TENSOR:
        {
            ast_node_t *node = alloc_node(AST_TENSOR_DECL);
            node->token = *advance(ps);
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            if (peek(ps) && peek(ps)->type == TOK_EQUAL) {
                advance(ps);
                node->children[1] = parse_expression(ps);
                node->child_count = 2;
            }
            return node;
        }

    case TOK_KW_TRAIN:
        {
            ast_node_t *node = alloc_node(AST_TRAIN);
            node->token = *advance(ps);
            /* Parse model name and data reference */
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            return node;
        }

    case TOK_KW_INFER:
        {
            ast_node_t *node = alloc_node(AST_INFER);
            node->token = *advance(ps);
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            return node;
        }

    case TOK_KW_LOAD:
        {
            ast_node_t *node = alloc_node(AST_LOAD);
            node->token = *advance(ps);
            node->children[0] = parse_expression(ps);
            node->child_count = 1;
            return node;
        }

    case TOK_KW_SAVE:
        {
            ast_node_t *node = alloc_node(AST_SAVE);
            node->token = *advance(ps);
            node->children[0] = parse_expression(ps);
            node->child_count = 1;
            return node;
        }

    case TOK_KW_GIT:
        {
            ast_node_t *node = alloc_node(AST_GIT_OP);
            node->token = *advance(ps);
            /* Parse git subcommand */
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            /* Parse message/args */
            if (peek(ps) && peek(ps)->type == TOK_STRING_LIT) {
                node->children[1] = alloc_node(AST_LITERAL);
                node->children[1]->token = *advance(ps);
                node->child_count = 2;
            }
            return node;
        }

    case TOK_KW_DEPLOY:
        {
            ast_node_t *node = alloc_node(AST_DEPLOY);
            node->token = *advance(ps);
            if (peek(ps) && peek(ps)->type == TOK_IDENT) {
                node->children[0] = alloc_node(AST_IDENT);
                node->children[0]->token = *advance(ps);
                node->child_count = 1;
            }
            return node;
        }

    case TOK_KW_PRINT:
        {
            ast_node_t *node = alloc_node(AST_PRINT);
            node->token = *advance(ps);
            node->children[0] = parse_expression(ps);
            node->child_count = 1;
            return node;
        }

    case TOK_IDENT:
        {
            /* Could be assignment: x = expr */
            ast_node_t *ident = alloc_node(AST_IDENT);
            ident->token = *advance(ps);
            if (peek(ps) && peek(ps)->type == TOK_EQUAL) {
                advance(ps);
                ast_node_t *node = alloc_node(AST_ASSIGNMENT);
                node->children[0] = ident;
                node->children[1] = parse_expression(ps);
                node->child_count = 2;
                return node;
            }
            return ident;
        }

    default:
        advance(ps); /* Skip unknown token */
        return parse_statement(ps);
    }
}

ast_node_t *pseudo_parse(const token_t *tokens, uint32_t token_count)
{
    parser_state_t ps = { tokens, token_count, 0 };
    ast_node_t *program = alloc_node(AST_PROGRAM);

    while (ps.pos < ps.count) {
        ast_node_t *stmt = parse_statement(&ps);
        if (stmt) {
            if (program->child_count < 8)
                program->children[program->child_count++] = stmt;
        }
        skip_newlines(&ps);
    }

    return program;
}

/* =============================================================================
 * IR Lowering - Convert AST to Tensor IR
 * =============================================================================*/

int pseudo_lower_to_ir(ast_node_t *ast, tir_program_t *program)
{
    if (!ast || !program) return -1;
    kmemset(program, 0, sizeof(*program));

    /* Walk AST and emit TIR instructions */
    for (uint32_t i = 0; i < ast->child_count; i++) {
        ast_node_t *stmt = ast->children[i];
        if (!stmt) continue;

        switch (stmt->type) {
        case AST_TENSOR_DECL:
            {
                tir_instruction_t *inst = &program->instructions[program->inst_count++];
                inst->opcode = TIR_ALLOC;
                inst->dest = program->reg_count++;
            }
            break;

        case AST_BINARY_OP:
            if (stmt->token.type == TOK_AT) {
                /* Matrix multiply */
                tir_instruction_t *inst = &program->instructions[program->inst_count++];
                inst->opcode = TIR_MATMUL;
                inst->dest = program->reg_count++;
                inst->num_operands = 2;
            }
            break;

        case AST_TRAIN:
            {
                /* Training is a complex operation that expands to many TIR instructions */
                tir_instruction_t *inst = &program->instructions[program->inst_count++];
                inst->opcode = TIR_FUSED_OP;
                inst->dest = program->reg_count++;
            }
            break;

        default:
            break;
        }
    }

    return 0;
}

/* =============================================================================
 * Optimization Passes
 * =============================================================================*/

int tir_optimize_fuse_ops(tir_program_t *program)
{
    /* Fuse common patterns:
     * matmul + bias_add + relu → fused_linear_relu
     * softmax(Q @ K^T / sqrt(d)) @ V → fused_attention
     * layernorm + dropout → fused_layernorm_dropout
     */
    uint32_t fusions = 0;

    for (uint32_t i = 0; i + 2 < program->inst_count; i++) {
        tir_instruction_t *a = &program->instructions[i];
        tir_instruction_t *b = &program->instructions[i + 1];
        tir_instruction_t *c = &program->instructions[i + 2];

        /* matmul + add + relu → fused */
        if (a->opcode == TIR_MATMUL && b->opcode == TIR_ADD &&
            (c->opcode == TIR_RELU || c->opcode == TIR_GELU)) {
            a->opcode = TIR_FUSED_OP;
            a->fused_next = b;
            b->fused_next = c;
            /* Mark b and c as NOP (they're part of the fused op) */
            b->opcode = TIR_NOP;
            c->opcode = TIR_NOP;
            fusions++;
            i += 2;
        }
    }

    kprintf_debug("[JIT] Fused %d operation groups\n", fusions);
    return fusions;
}

int tir_optimize_precision(tir_program_t *program)
{
    /* Auto-tune precision:
     * - Attention: can use FP16 for Q,K but FP32 for softmax
     * - MatMul: FP16 or BF16 with FP32 accumulation
     * - LayerNorm: keep FP32
     */
    uint32_t downgrades = 0;

    for (uint32_t i = 0; i < program->inst_count; i++) {
        tir_instruction_t *inst = &program->instructions[i];
        if (inst->opcode == TIR_NOP) continue;

        if (inst->dtype == TENSOR_DTYPE_F32) {
            switch (inst->opcode) {
            case TIR_MATMUL:
            case TIR_ATTENTION:
            case TIR_EMBEDDING:
                inst->dtype = TENSOR_DTYPE_F16; /* Safe to downgrade */
                downgrades++;
                break;
            case TIR_LAYERNORM:
            case TIR_SOFTMAX:
                /* Keep FP32 for numerical stability */
                break;
            default:
                break;
            }
        }
    }

    kprintf_debug("[JIT] Precision optimized %d operations to FP16\n", downgrades);
    return downgrades;
}

/* =============================================================================
 * Interpreter (used before JIT kicks in)
 * =============================================================================*/

static runtime_value_t interpret_node(pseudo_runtime_t *rt, ast_node_t *node)
{
    runtime_value_t result = { .type = VAL_NONE };
    if (!node) return result;

    switch (node->type) {
    case AST_LITERAL:
        switch (node->token.type) {
        case TOK_INT_LIT:
            result.type = VAL_INT;
            result.int_val = node->token.value.int_val;
            break;
        case TOK_FLOAT_LIT:
            result.type = VAL_FLOAT;
            result.float_val = node->token.value.float_val;
            break;
        case TOK_BOOL_LIT:
            result.type = VAL_BOOL;
            result.bool_val = node->token.value.bool_val;
            break;
        case TOK_STRING_LIT:
            result.type = VAL_STRING;
            result.string_val = (char *)node->token.start;
            break;
        default:
            break;
        }
        break;

    case AST_IDENT:
        /* Look up variable */
        for (uint32_t i = 0; i < rt->var_count; i++) {
            bool match = true;
            for (uint32_t j = 0; j < node->token.length; j++) {
                if (rt->vars[i].name[j] != node->token.start[j]) {
                    match = false;
                    break;
                }
            }
            if (match && rt->vars[i].name[node->token.length] == '\0') {
                return rt->vars[i].value;
            }
        }
        break;

    case AST_ASSIGNMENT:
        {
            runtime_value_t val = interpret_node(rt, node->children[1]);
            /* Store variable */
            if (rt->var_count < PSEUDO_MAX_VARS && node->children[0]) {
                variable_t *var = &rt->vars[rt->var_count++];
                for (uint32_t i = 0; i < node->children[0]->token.length && i < 63; i++)
                    var->name[i] = node->children[0]->token.start[i];
                var->value = val;
            }
            result = val;
        }
        break;

    case AST_BINARY_OP:
        {
            runtime_value_t left = interpret_node(rt, node->children[0]);
            runtime_value_t right = interpret_node(rt, node->children[1]);

            if (left.type == VAL_INT && right.type == VAL_INT) {
                result.type = VAL_INT;
                switch (node->token.type) {
                case TOK_PLUS:  result.int_val = left.int_val + right.int_val; break;
                case TOK_MINUS: result.int_val = left.int_val - right.int_val; break;
                case TOK_STAR:  result.int_val = left.int_val * right.int_val; break;
                case TOK_SLASH: result.int_val = right.int_val ? left.int_val / right.int_val : 0; break;
                default: break;
                }
            }

            if (node->token.type == TOK_AT) {
                /* Matrix multiply - dispatch to GPU */
                result.type = VAL_TENSOR;
                /* TODO: actual tensor matmul via scheduler */
                rt->ops_executed++;
            }
        }
        break;

    case AST_PRINT:
        {
            runtime_value_t val = interpret_node(rt, node->children[0]);
            switch (val.type) {
            case VAL_INT:    kprintf("%ld\n", val.int_val); break;
            case VAL_FLOAT:  kprintf("%f\n", val.float_val); break;
            case VAL_BOOL:   kprintf("%s\n", val.bool_val ? "true" : "false"); break;
            case VAL_STRING: kprintf("%s\n", val.string_val); break;
            default:         kprintf("<value>\n"); break;
            }
        }
        break;

    case AST_PROGRAM:
        for (uint32_t i = 0; i < node->child_count; i++) {
            result = interpret_node(rt, node->children[i]);
        }
        break;

    case AST_GIT_OP:
        /* Git operations */
        kprintf("[GIT] Operation from pseudocode\n");
        /* TODO: dispatch to kernel git subsystem */
        break;

    case AST_TRAIN:
        kprintf("[TRAIN] Starting training...\n");
        break;

    case AST_INFER:
        kprintf("[INFER] Running inference...\n");
        break;

    case AST_DEPLOY:
        kprintf("[DEPLOY] Deploying model...\n");
        break;

    default:
        break;
    }

    rt->ops_executed++;
    return result;
}

int pseudo_interpret(pseudo_runtime_t *rt, ast_node_t *ast,
                      runtime_value_t *result)
{
    runtime_value_t val = interpret_node(rt, ast);
    if (result) *result = val;
    return 0;
}

/* =============================================================================
 * High-Level Execution
 * =============================================================================*/

int pseudo_exec_string(pseudo_runtime_t *rt, const char *source)
{
    /* Lex */
    token_t tokens[1024];
    uint32_t token_count;
    int ret = pseudo_lex(source, tokens, 1024, &token_count);
    if (ret != 0) return ret;

    /* Parse */
    ast_node_t *ast = pseudo_parse(tokens, token_count);
    if (!ast) return -1;

    /* Check if any function has been called enough times for JIT */
    /* For now, always interpret */
    runtime_value_t result;
    ret = pseudo_interpret(rt, ast, &result);

    /* TODO: free AST */
    return ret;
}
