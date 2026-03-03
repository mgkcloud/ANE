// btc_mlp.m — Train a simple MLP on BTC signal data using ANE
// Architecture: 15 features → 64 hidden (ReLU) → 1 output (sigmoid)
// Fixes: checkpoint restore on exec(), class-weighted loss, inference benchmark
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define IN_DIM 20
#define HIDDEN 128
#define OUT_DIM 1
#define BATCH_SIZE 32
#define LR 0.001f
#define MAX_COMPILES 130  // conservative for M4
#define ACCUM_STEPS 50
#define TOTAL_EPOCHS 50
#define CKPT_PATH "btc_mlp_ckpt.bin"
#define CKPT_MAGIC 0x4D4C5043

typedef struct { uint32_t magic, version, n_samples, n_features; } DataHeader;

// Checkpoint layout
typedef struct {
    int32_t magic;
    int32_t start_epoch;  // next epoch to run
    int32_t adam_t;
    // followed by: W1, W2, b1, b2, mW1, vW1, mW2, vW2, mb1, vb1, mb2, vb2
} CkptHeader;

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
}

static NSData *build_weight_blob(const float *w, int rows, int cols) {
    int ws = rows * cols * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (int i = 0; i < rows * cols; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *gen_conv_mil(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cin\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)"
        "[name = string(\"conv\")];\n"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = y16)[name = string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", in_ch, spatial, in_ch, spatial, out_ch, in_ch, out_ch, in_ch, out_ch, spatial, out_ch, spatial];
}

typedef struct {
    void *model;
    IOSurfaceRef ioIn, ioOut;
    void *request;
    void *tmpDir;
} Kern;

static Kern *compile_conv(int in_ch, int out_ch, int spatial, const float *weights) {
    NSString *mil = gen_conv_mil(in_ch, out_ch, spatial);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData *wdata = build_weight_blob(weights, out_ch, in_ch);

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (e) { NSLog(@"Compile error: %@", e); return NULL; }
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (e) { NSLog(@"Load error: %@", e); return NULL; }
    g_compile_count++;

    IOSurfaceRef ioIn = make_surface(in_ch * spatial * 4);
    IOSurfaceRef ioOut = make_surface(out_ch * spatial * 4);

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = CFBridgingRetain(mdl);
    k->ioIn = ioIn; k->ioOut = ioOut;
    k->request = CFBridgingRetain(req);
    k->tmpDir = CFBridgingRetain(td);
    return k;
}

static void ane_eval(Kern *k) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        (__bridge id)(k->model), @selector(evaluateWithQoS:options:request:error:),
        21, @{}, (__bridge id)(k->request), &e);
}

static void free_kern(Kern *k) {
    if (!k) return;
    if (k->tmpDir) {
        [[NSFileManager defaultManager] removeItemAtPath:(__bridge_transfer NSString*)(k->tmpDir) error:nil];
        k->tmpDir = NULL;
    }
    if (k->request) { CFRelease(k->request); k->request = NULL; }
    if (k->model) { CFRelease(k->model); k->model = NULL; }
    if (k->ioIn) CFRelease(k->ioIn);
    if (k->ioOut) CFRelease(k->ioOut);
    free(k);
}

static void save_checkpoint(int next_epoch, int adam_t,
    float *W1, float *W2, float *b1, float *b2,
    float *mW1, float *vW1, float *mW2, float *vW2,
    float *mb1, float *vb1, float *mb2, float *vb2) {
    FILE *f = fopen(CKPT_PATH, "wb");
    if (!f) return;
    CkptHeader h = { .magic = CKPT_MAGIC, .start_epoch = next_epoch, .adam_t = adam_t };
    fwrite(&h, sizeof(h), 1, f);
    fwrite(W1, 4, HIDDEN*IN_DIM, f);
    fwrite(W2, 4, OUT_DIM*HIDDEN, f);
    fwrite(b1, 4, HIDDEN, f);
    fwrite(b2, 4, OUT_DIM, f);
    fwrite(mW1, 4, HIDDEN*IN_DIM, f); fwrite(vW1, 4, HIDDEN*IN_DIM, f);
    fwrite(mW2, 4, OUT_DIM*HIDDEN, f); fwrite(vW2, 4, OUT_DIM*HIDDEN, f);
    fwrite(mb1, 4, HIDDEN, f); fwrite(vb1, 4, HIDDEN, f);
    fwrite(mb2, 4, OUT_DIM, f); fwrite(vb2, 4, OUT_DIM, f);
    fclose(f);
}

static int load_checkpoint(int *start_epoch, int *adam_t,
    float *W1, float *W2, float *b1, float *b2,
    float *mW1, float *vW1, float *mW2, float *vW2,
    float *mb1, float *vb1, float *mb2, float *vb2) {
    FILE *f = fopen(CKPT_PATH, "rb");
    if (!f) return 0;
    CkptHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != CKPT_MAGIC) { fclose(f); return 0; }
    *start_epoch = h.start_epoch;
    *adam_t = h.adam_t;
    fread(W1, 4, HIDDEN*IN_DIM, f);
    fread(W2, 4, OUT_DIM*HIDDEN, f);
    fread(b1, 4, HIDDEN, f);
    fread(b2, 4, OUT_DIM, f);
    fread(mW1, 4, HIDDEN*IN_DIM, f); fread(vW1, 4, HIDDEN*IN_DIM, f);
    fread(mW2, 4, OUT_DIM*HIDDEN, f); fread(vW2, 4, OUT_DIM*HIDDEN, f);
    fread(mb1, 4, HIDDEN, f); fread(vb1, 4, HIDDEN, f);
    fread(mb2, 4, OUT_DIM, f); fread(vb2, 4, OUT_DIM, f);
    fclose(f);
    return 1;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        const char *data_path = argc > 1 ? argv[1] : "btc_training_data.bin";
        int resuming = (argc > 2 && strcmp(argv[2], "--resume") == 0);

        // Load data
        int fd = open(data_path, O_RDONLY);
        if (fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(fd, &st);
        void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        DataHeader *hdr = (DataHeader*)data;
        if (hdr->magic != 0x42544353) { printf("Bad magic\n"); return 1; }

        int n_samples = hdr->n_samples;
        int n_features = hdr->n_features;
        float *samples = (float*)((uint8_t*)data + 16);

        // Count class distribution for weighted loss
        int n_pos = 0, n_neg = 0;
        for (int i = 0; i < n_samples; i++) {
            if (samples[i * (n_features+1) + n_features] > 0.5f) n_pos++; else n_neg++;
        }
        // Weight: inverse frequency. Positive weight = n_neg/n_pos (< 1 since pos is majority)
        float w_pos = (float)n_neg / (float)n_pos;
        float w_neg = 1.0f;
        // Normalize so mean weight = 1
        float w_mean = (w_pos * n_pos + w_neg * n_neg) / n_samples;
        w_pos /= w_mean;
        w_neg /= w_mean;

        printf("=== BTC Signal MLP on ANE ===\n");
        printf("Data: %d samples (%d pos / %d neg), %d features\n", n_samples, n_pos, n_neg, n_features);
        printf("Class weights: pos=%.3f neg=%.3f\n", w_pos, w_neg);
        printf("Architecture: %d -> %d (ReLU) -> %d (sigmoid)\n", IN_DIM, HIDDEN, OUT_DIM);

        // Allocate weights + Adam state
        float *W1 = (float*)calloc(HIDDEN * IN_DIM, 4);
        float *W2 = (float*)calloc(OUT_DIM * HIDDEN, 4);
        float *b1 = (float*)calloc(HIDDEN, 4);
        float *b2 = (float*)calloc(OUT_DIM, 4);
        float *mW1 = (float*)calloc(HIDDEN*IN_DIM, 4), *vW1 = (float*)calloc(HIDDEN*IN_DIM, 4);
        float *mW2 = (float*)calloc(OUT_DIM*HIDDEN, 4), *vW2 = (float*)calloc(OUT_DIM*HIDDEN, 4);
        float *mb1 = (float*)calloc(HIDDEN, 4), *vb1 = (float*)calloc(HIDDEN, 4);
        float *mb2 = (float*)calloc(OUT_DIM, 4), *vb2 = (float*)calloc(OUT_DIM, 4);
        float *gW1 = (float*)calloc(HIDDEN*IN_DIM, 4), *gW2 = (float*)calloc(OUT_DIM*HIDDEN, 4);
        float *gb1 = (float*)calloc(HIDDEN, 4), *gb2 = (float*)calloc(OUT_DIM, 4);
        float *h = (float*)calloc(HIDDEN * BATCH_SIZE, 4);
        float *h_relu = (float*)calloc(HIDDEN * BATCH_SIZE, 4);
        float *out_buf = (float*)calloc(BATCH_SIZE, 4);

        int start_epoch = 0, adam_t = 0;

        if (resuming && load_checkpoint(&start_epoch, &adam_t,
                W1, W2, b1, b2, mW1, vW1, mW2, vW2, mb1, vb1, mb2, vb2)) {
            printf("[RESUMED from checkpoint at epoch %d, adam_t=%d]\n", start_epoch, adam_t);
        } else {
            // Xavier init
            srand48(42);
            float s1 = sqrtf(2.0f / IN_DIM), s2 = sqrtf(2.0f / HIDDEN);
            for (int i = 0; i < HIDDEN * IN_DIM; i++) W1[i] = s1 * (2*drand48()-1);
            for (int i = 0; i < OUT_DIM * HIDDEN; i++) W2[i] = s2 * (2*drand48()-1);
            printf("[Fresh init]\n");
        }

        if (start_epoch >= TOTAL_EPOCHS) {
            printf("Already completed %d epochs. Skipping to eval.\n", TOTAL_EPOCHS);
            goto eval_phase;
        }

        // Compile initial kernels
        printf("Compiling layers...\n");
        Kern *k1 = compile_conv(IN_DIM, HIDDEN, BATCH_SIZE, W1);
        Kern *k2 = compile_conv(HIDDEN, OUT_DIM, BATCH_SIZE, W2);
        if (!k1 || !k2) { printf("Compile failed\n"); return 1; }
        printf("Ready (compiles=%d)\n\n", g_compile_count);

        float b1_adam = 0.9f, b2_adam = 0.999f, eps = 1e-8f;

        for (int epoch = start_epoch; epoch < TOTAL_EPOCHS; epoch++) {
            float epoch_loss = 0;
            int correct = 0, total = 0;
            // Track per-class accuracy
            int tp = 0, tn = 0, fp = 0, fn = 0;
            uint64_t t_epoch = mach_absolute_time();

            // Shuffle
            int *indices = (int*)malloc(n_samples * sizeof(int));
            for (int i = 0; i < n_samples; i++) indices[i] = i;
            srand48(42 + epoch);  // deterministic per epoch
            for (int i = n_samples-1; i > 0; i--) {
                int j = (int)(drand48() * (i+1));
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }

            memset(gW1, 0, HIDDEN*IN_DIM*4);
            memset(gW2, 0, OUT_DIM*HIDDEN*4);
            memset(gb1, 0, HIDDEN*4);
            memset(gb2, 0, OUT_DIM*4);
            int accum = 0;

            for (int step = 0; step + BATCH_SIZE <= n_samples; step += BATCH_SIZE) {
                // Forward: load batch into L1 input
                IOSurfaceLock(k1->ioIn, 0, NULL);
                float *inp = (float*)IOSurfaceGetBaseAddress(k1->ioIn);
                for (int b = 0; b < BATCH_SIZE; b++) {
                    float *sample = samples + indices[step+b] * (n_features+1);
                    for (int f = 0; f < IN_DIM; f++)
                        inp[f * BATCH_SIZE + b] = sample[f];
                }
                IOSurfaceUnlock(k1->ioIn, 0, NULL);

                ane_eval(k1);  // L1 matmul on ANE

                // ReLU + bias on CPU
                IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *h_out = (float*)IOSurfaceGetBaseAddress(k1->ioOut);
                for (int j = 0; j < HIDDEN; j++)
                    for (int b = 0; b < BATCH_SIZE; b++) {
                        float v = h_out[j*BATCH_SIZE+b] + b1[j];
                        h[j*BATCH_SIZE+b] = v;
                        h_relu[j*BATCH_SIZE+b] = v > 0 ? v : 0;
                    }
                IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);

                IOSurfaceLock(k2->ioIn, 0, NULL);
                memcpy(IOSurfaceGetBaseAddress(k2->ioIn), h_relu, HIDDEN*BATCH_SIZE*4);
                IOSurfaceUnlock(k2->ioIn, 0, NULL);

                ane_eval(k2);  // L2 matmul on ANE

                // Sigmoid + weighted loss on CPU
                IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *out_raw = (float*)IOSurfaceGetBaseAddress(k2->ioOut);

                float dout[BATCH_SIZE];
                for (int b = 0; b < BATCH_SIZE; b++) {
                    float logit = out_raw[b] + b2[0];
                    float pred = 1.0f / (1.0f + expf(-logit));
                    out_buf[b] = pred;

                    float label = samples[indices[step+b]*(n_features+1)+n_features];
                    float w = label > 0.5f ? w_pos : w_neg;

                    // Weighted BCE
                    float p = fmaxf(fminf(pred, 1-1e-7f), 1e-7f);
                    epoch_loss += w * -(label * logf(p) + (1-label) * logf(1-p));

                    int pred_cls = pred > 0.5f;
                    int true_cls = label > 0.5f;
                    if (pred_cls && true_cls) tp++;
                    else if (!pred_cls && !true_cls) tn++;
                    else if (pred_cls && !true_cls) fp++;
                    else fn++;
                    if (pred_cls == true_cls) correct++;
                    total++;

                    // Gradient: weighted (pred - label)
                    dout[b] = w * (pred - label);
                }
                IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Backward
                for (int b = 0; b < BATCH_SIZE; b++) {
                    gb2[0] += dout[b];
                    for (int j = 0; j < HIDDEN; j++)
                        gW2[j] += dout[b] * h_relu[j*BATCH_SIZE+b];
                }

                float dh_relu[HIDDEN * BATCH_SIZE];
                for (int b = 0; b < BATCH_SIZE; b++)
                    for (int j = 0; j < HIDDEN; j++)
                        dh_relu[j*BATCH_SIZE+b] = W2[j] * dout[b];

                for (int b = 0; b < BATCH_SIZE; b++) {
                    float *x = samples + indices[step+b]*(n_features+1);
                    for (int j = 0; j < HIDDEN; j++) {
                        float dh = h[j*BATCH_SIZE+b] > 0 ? dh_relu[j*BATCH_SIZE+b] : 0;
                        gb1[j] += dh;
                        for (int f = 0; f < IN_DIM; f++)
                            gW1[j*IN_DIM+f] += dh * x[f];
                    }
                }

                accum++;
                if (accum >= ACCUM_STEPS) {
                    adam_t++;
                    float scale = 1.0f / (accum * BATCH_SIZE);

                    #define ADAM(w, g, m, v, sz) do { \
                        for (int _i = 0; _i < (sz); _i++) { \
                            float gi = (g)[_i] * scale; \
                            (m)[_i] = b1_adam*(m)[_i] + (1-b1_adam)*gi; \
                            (v)[_i] = b2_adam*(v)[_i] + (1-b2_adam)*gi*gi; \
                            float mh = (m)[_i]/(1-powf(b1_adam,adam_t)); \
                            float vh = (v)[_i]/(1-powf(b2_adam,adam_t)); \
                            (w)[_i] -= LR * mh / (sqrtf(vh)+eps); \
                        } \
                    } while(0)

                    ADAM(W1, gW1, mW1, vW1, HIDDEN*IN_DIM);
                    ADAM(W2, gW2, mW2, vW2, OUT_DIM*HIDDEN);
                    ADAM(b1, gb1, mb1, vb1, HIDDEN);
                    ADAM(b2, gb2, mb2, vb2, OUT_DIM);
                    #undef ADAM

                    memset(gW1, 0, HIDDEN*IN_DIM*4);
                    memset(gW2, 0, OUT_DIM*HIDDEN*4);
                    memset(gb1, 0, HIDDEN*4);
                    memset(gb2, 0, OUT_DIM*4);
                    accum = 0;

                    // Recompile with updated weights
                    if (g_compile_count + 2 <= MAX_COMPILES) {
                        free_kern(k1); free_kern(k2);
                        k1 = compile_conv(IN_DIM, HIDDEN, BATCH_SIZE, W1);
                        k2 = compile_conv(HIDDEN, OUT_DIM, BATCH_SIZE, W2);
                    }
                }
            }

            free(indices);
            double epoch_ms = tb_ms(mach_absolute_time() - t_epoch);
            float acc = 100.0f * correct / total;
            float precision = tp > 0 ? 100.0f*tp/(tp+fp) : 0;
            float recall = tp > 0 ? 100.0f*tp/(tp+fn) : 0;
            float neg_acc = (tn+fn) > 0 ? 100.0f*tn/(tn+fp) : 0;
            printf("Epoch %2d: loss=%.4f acc=%.1f%% P=%.1f%% R=%.1f%% TP=%d TN=%d FP=%d FN=%d  %.0fms c=%d\n",
                   epoch, epoch_loss/total, acc, precision, recall, tp, tn, fp, fn, epoch_ms, g_compile_count);

            // Check compile budget - need room for 2 compiles per accumulation step
            // Each epoch uses ~11 weight updates = 22 compiles
            if (g_compile_count + 24 > MAX_COMPILES) {
                save_checkpoint(epoch+1, adam_t, W1, W2, b1, b2,
                    mW1, vW1, mW2, vW2, mb1, vb1, mb2, vb2);
                printf("[exec() restart after epoch %d, compiles=%d]\n", epoch, g_compile_count);
                free_kern(k1); free_kern(k2);
                execl(argv[0], argv[0], data_path, "--resume", NULL);
                perror("execl"); return 1;
            }
        }

                // Save final
        save_checkpoint(TOTAL_EPOCHS, adam_t, W1, W2, b1, b2,
            mW1, vW1, mW2, vW2, mb1, vb1, mb2, vb2);
        free_kern(k1); free_kern(k2);

eval_phase:;
        // Final eval + inference benchmark
        printf("\n=== Final Evaluation ===\n");
        Kern *ek1 = compile_conv(IN_DIM, HIDDEN, BATCH_SIZE, W1);
        Kern *ek2 = compile_conv(HIDDEN, OUT_DIM, BATCH_SIZE, W2);
        if (!ek1 || !ek2) { printf("Eval compile failed\n"); return 1; }

        int tp2=0, tn2=0, fp2=0, fn2=0;
        for (int i = 0; i + BATCH_SIZE <= n_samples; i += BATCH_SIZE) {
            IOSurfaceLock(ek1->ioIn, 0, NULL);
            float *inp = (float*)IOSurfaceGetBaseAddress(ek1->ioIn);
            for (int b = 0; b < BATCH_SIZE; b++) {
                float *s = samples + (i+b)*(n_features+1);
                for (int f = 0; f < IN_DIM; f++) inp[f*BATCH_SIZE+b] = s[f];
            }
            IOSurfaceUnlock(ek1->ioIn, 0, NULL);
            ane_eval(ek1);

            IOSurfaceLock(ek1->ioOut, kIOSurfaceLockReadOnly, NULL);
            float *ho = (float*)IOSurfaceGetBaseAddress(ek1->ioOut);
            IOSurfaceLock(ek2->ioIn, 0, NULL);
            float *h2 = (float*)IOSurfaceGetBaseAddress(ek2->ioIn);
            for (int j = 0; j < HIDDEN*BATCH_SIZE; j++)
                h2[j] = fmaxf(ho[j] + b1[j/BATCH_SIZE], 0);
            IOSurfaceUnlock(ek1->ioOut, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(ek2->ioIn, 0, NULL);

            ane_eval(ek2);

            IOSurfaceLock(ek2->ioOut, kIOSurfaceLockReadOnly, NULL);
            float *oo = (float*)IOSurfaceGetBaseAddress(ek2->ioOut);
            for (int b = 0; b < BATCH_SIZE; b++) {
                float pred = 1.0f/(1.0f+expf(-(oo[b]+b2[0])));
                float label = samples[(i+b)*(n_features+1)+n_features];
                int pc = pred > 0.5f, tc = label > 0.5f;
                if (pc && tc) tp2++;
                else if (!pc && !tc) tn2++;
                else if (pc && !tc) fp2++;
                else fn2++;
            }
            IOSurfaceUnlock(ek2->ioOut, kIOSurfaceLockReadOnly, NULL);
        }
        int total2 = tp2+tn2+fp2+fn2;
        printf("Accuracy: %.1f%% (%d/%d)\n", 100.0f*(tp2+tn2)/total2, tp2+tn2, total2);
        printf("Precision: %.1f%%  Recall: %.1f%%\n",
            tp2 > 0 ? 100.0f*tp2/(tp2+fp2) : 0,
            tp2 > 0 ? 100.0f*tp2/(tp2+fn2) : 0);
        printf("Neg accuracy: %.1f%% (TN=%d FP=%d)\n",
            (tn2+fp2) > 0 ? 100.0f*tn2/(tn2+fp2) : 0, tn2, fp2);
        printf("TP=%d TN=%d FP=%d FN=%d\n", tp2, tn2, fp2, fn2);

        // Inference speed
        printf("\nInference benchmark (1000 forward passes)...\n");
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 1000; i++) { ane_eval(ek1); ane_eval(ek2); }
        double ms = tb_ms(mach_absolute_time() - t0);
        printf("Total: %.1fms  Per-inference: %.3fms (%.0f inferences/sec)\n",
            ms, ms/1000.0, 1000000.0/ms);

        // Save production model
        FILE *mf = fopen("btc_mlp_final.bin", "wb");
        if (mf) {
            int magic = 0x4D4C5046;
            fwrite(&magic, 4, 1, mf);
            int dims[] = {IN_DIM, HIDDEN, OUT_DIM};
            fwrite(dims, 4, 3, mf);
            fwrite(W1, 4, HIDDEN*IN_DIM, mf);
            fwrite(W2, 4, OUT_DIM*HIDDEN, mf);
            fwrite(b1, 4, HIDDEN, mf);
            fwrite(b2, 4, OUT_DIM, mf);
            fclose(mf);
            printf("\nModel saved to btc_mlp_final.bin\n");
        }

        free_kern(ek1); free_kern(ek2);
        munmap(data, st.st_size); close(fd);
        free(W1); free(W2); free(b1); free(b2);
        free(mW1); free(vW1); free(mW2); free(vW2);
        free(mb1); free(vb1); free(mb2); free(vb2);
        free(gW1); free(gW2); free(gb1); free(gb2);
        free(h); free(h_relu); free(out_buf);
        printf("Done.\n");
    }
    return 0;
}