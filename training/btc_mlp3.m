// btc_mlp3.m — 3-layer MLP on ANE for BTC signal prediction
// Architecture: 20 → 128 (ReLU) → 64 (ReLU) → 1 (sigmoid)
// With: gradient clipping, LR warmup+decay, label smoothing
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
#define H1 128
#define H2 64
#define OUT_DIM 1
#define BATCH_SIZE 32
#define BASE_LR 0.001f
#define MAX_COMPILES 130
#define ACCUM_STEPS 50
#define TOTAL_EPOCHS 80
#define WARMUP_EPOCHS 3
#define GRAD_CLIP 1.0f
#define LABEL_SMOOTH 0.05f
#define CKPT_PATH "btc_mlp3_ckpt.bin"
#define CKPT_MAGIC 0x4D4C5033

// Total weight count for checkpoint
#define W1_SZ (H1*IN_DIM)
#define W2_SZ (H2*H1)
#define W3_SZ (OUT_DIM*H2)
#define B1_SZ H1
#define B2_SZ H2
#define B3_SZ OUT_DIM
#define TOTAL_PARAMS (W1_SZ+W2_SZ+W3_SZ+B1_SZ+B2_SZ+B3_SZ)

typedef struct { uint32_t magic, version, n_samples, n_features; } DataHeader;
typedef struct { int32_t magic, start_epoch, adam_t, pad; } CkptHeader;

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_cc = 0;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef mk_surf(size_t b) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b),(id)kIOSurfacePixelFormat:@0});
}

static NSData *mk_blob(const float *w, int r, int c) {
    int ws = r*c*2, tot = 128+ws;
    uint8_t *buf = (uint8_t*)calloc(tot, 1);
    buf[0]=1; buf[4]=2;
    buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72) = ws; *(uint32_t*)(buf+80) = 128;
    _Float16 *fp = (_Float16*)(buf+128);
    for (int i = 0; i < r*c; i++) fp[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:tot freeWhenDone:YES];
}

static NSString *mil(int ic, int oc, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
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
        "    } -> (y);\n}\n", ic, sp, ic, sp, oc, ic, oc, ic, oc, sp, oc, sp];
}

typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request, *tmpDir; } K;

static K *comp(int ic, int oc, int sp, const float *w) {
    NSString *m = mil(ic, oc, sp);
    NSData *md = [m dataUsingEncoding:NSUTF8StringEncoding];
    NSData *wd = mk_blob(w, oc, ic);
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wd}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wd writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (e) { NSLog(@"C err: %@", e); return NULL; }
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (e) { NSLog(@"L err: %@", e); return NULL; }
    g_cc++;
    IOSurfaceRef ii = mk_surf(ic*sp*4), io = mk_surf(oc*sp*4);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ii);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), io);
    id rq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    K *k = (K*)calloc(1, sizeof(K));
    k->model = (void*)CFBridgingRetain(mdl); k->ioIn = ii; k->ioOut = io;
    k->request = (void*)CFBridgingRetain(rq); k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
}

static void eval(K *k) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        (__bridge id)(k->model), @selector(evaluateWithQoS:options:request:error:),
        21, @{}, (__bridge id)(k->request), &e);
}

static void freeK(K *k) {
    if (!k) return;
    if (k->tmpDir) { [[NSFileManager defaultManager] removeItemAtPath:(__bridge_transfer NSString*)(k->tmpDir) error:nil]; }
    if (k->request) CFRelease(k->request);
    if (k->model) CFRelease(k->model);
    if (k->ioIn) CFRelease(k->ioIn);
    if (k->ioOut) CFRelease(k->ioOut);
    free(k);
}

static float clip(float x) { return x > GRAD_CLIP ? GRAD_CLIP : (x < -GRAD_CLIP ? -GRAD_CLIP : x); }

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        const char *dp = argc > 1 ? argv[1] : "btc_training_v2.bin";
        int resuming = (argc > 2 && strcmp(argv[2], "--resume") == 0);

        int fd = open(dp, O_RDONLY);
        if (fd < 0) { printf("Cannot open %s\n", dp); return 1; }
        struct stat st; fstat(fd, &st);
        void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        DataHeader *hdr = (DataHeader*)data;
        if (hdr->magic != 0x42544353) { printf("Bad magic\n"); return 1; }
        int ns = hdr->n_samples, nf = hdr->n_features;
        float *S = (float*)((uint8_t*)data + 16);

        int np = 0, nn = 0;
        for (int i = 0; i < ns; i++) { if (S[i*(nf+1)+nf] > 0.5f) np++; else nn++; }
        float wp = (float)nn/np, wn = 1.0f;
        float wm = (wp*np + wn*nn)/ns; wp /= wm; wn /= wm;

        printf("=== BTC 3-Layer MLP on ANE ===\n");
        printf("Data: %d samples (%d pos / %d neg), %d features\n", ns, np, nn, nf);
        printf("Arch: %d -> %d -> %d -> %d | LR=%.4f | clip=%.1f | smooth=%.2f\n",
            IN_DIM, H1, H2, OUT_DIM, BASE_LR, GRAD_CLIP, LABEL_SMOOTH);

        // Weights
        float *W1 = calloc(W1_SZ, 4), *W2 = calloc(W2_SZ, 4), *W3 = calloc(W3_SZ, 4);
        float *b1 = calloc(B1_SZ, 4), *b2 = calloc(B2_SZ, 4), *b3 = calloc(B3_SZ, 4);
        // Adam momentum/variance for each
        float *mW1 = calloc(W1_SZ, 4), *vW1 = calloc(W1_SZ, 4);
        float *mW2 = calloc(W2_SZ, 4), *vW2 = calloc(W2_SZ, 4);
        float *mW3 = calloc(W3_SZ, 4), *vW3 = calloc(W3_SZ, 4);
        float *mb1 = calloc(B1_SZ, 4), *vb1 = calloc(B1_SZ, 4);
        float *mb2 = calloc(B2_SZ, 4), *vb2 = calloc(B2_SZ, 4);
        float *mb3 = calloc(B3_SZ, 4), *vb3 = calloc(B3_SZ, 4);
        // Gradients
        float *gW1 = calloc(W1_SZ, 4), *gW2 = calloc(W2_SZ, 4), *gW3 = calloc(W3_SZ, 4);
        float *gb1 = calloc(B1_SZ, 4), *gb2 = calloc(B2_SZ, 4), *gb3 = calloc(B3_SZ, 4);
        // Activations (per batch)
        float *a1 = calloc(H1*BATCH_SIZE, 4), *r1 = calloc(H1*BATCH_SIZE, 4);
        float *a2 = calloc(H2*BATCH_SIZE, 4), *r2 = calloc(H2*BATCH_SIZE, 4);
        float *out = calloc(BATCH_SIZE, 4);

        int se = 0, at = 0;

        if (resuming) {
            FILE *f = fopen(CKPT_PATH, "rb");
            if (f) {
                CkptHeader ch; fread(&ch, sizeof(ch), 1, f);
                if (ch.magic == CKPT_MAGIC) {
                    se = ch.start_epoch; at = ch.adam_t;
                    fread(W1, 4, W1_SZ, f); fread(W2, 4, W2_SZ, f); fread(W3, 4, W3_SZ, f);
                    fread(b1, 4, B1_SZ, f); fread(b2, 4, B2_SZ, f); fread(b3, 4, B3_SZ, f);
                    fread(mW1, 4, W1_SZ, f); fread(vW1, 4, W1_SZ, f);
                    fread(mW2, 4, W2_SZ, f); fread(vW2, 4, W2_SZ, f);
                    fread(mW3, 4, W3_SZ, f); fread(vW3, 4, W3_SZ, f);
                    fread(mb1, 4, B1_SZ, f); fread(vb1, 4, B1_SZ, f);
                    fread(mb2, 4, B2_SZ, f); fread(vb2, 4, B2_SZ, f);
                    fread(mb3, 4, B3_SZ, f); fread(vb3, 4, B3_SZ, f);
                    printf("[RESUMED epoch %d, adam_t=%d]\n", se, at);
                }
                fclose(f);
            }
        }
        if (!resuming || se == 0) {
            srand48(42);
            float s1 = sqrtf(2.0f/IN_DIM), s2 = sqrtf(2.0f/H1), s3 = sqrtf(2.0f/H2);
            for (int i = 0; i < W1_SZ; i++) W1[i] = s1*(2*drand48()-1);
            for (int i = 0; i < W2_SZ; i++) W2[i] = s2*(2*drand48()-1);
            for (int i = 0; i < W3_SZ; i++) W3[i] = s3*(2*drand48()-1);
            if (se == 0) printf("[Fresh init]\n");
        }

        if (se >= TOTAL_EPOCHS) goto eval_only;

        {
        K *k1 = comp(IN_DIM, H1, BATCH_SIZE, W1);
        K *k2 = comp(H1, H2, BATCH_SIZE, W2);
        K *k3 = comp(H2, OUT_DIM, BATCH_SIZE, W3);
        if (!k1||!k2||!k3) { printf("Compile fail\n"); return 1; }
        printf("Compiled (c=%d)\n\n", g_cc);

        float ba = 0.9f, bb = 0.999f, eps = 1e-8f;

        for (int ep = se; ep < TOTAL_EPOCHS; ep++) {
            // LR schedule: warmup then cosine decay
            float lr = BASE_LR;
            if (ep < WARMUP_EPOCHS) lr = BASE_LR * (ep + 1.0f) / WARMUP_EPOCHS;
            else {
                float prog = (float)(ep - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS);
                lr = BASE_LR * 0.5f * (1.0f + cosf(M_PI * prog));
            }

            float eloss = 0; int tp=0, tn=0, fp=0, fn=0;
            uint64_t t0 = mach_absolute_time();

            int *idx = malloc(ns * 4);
            for (int i = 0; i < ns; i++) idx[i] = i;
            srand48(42 + ep);
            for (int i = ns-1; i > 0; i--) { int j = drand48()*(i+1); int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }

            memset(gW1, 0, W1_SZ*4); memset(gW2, 0, W2_SZ*4); memset(gW3, 0, W3_SZ*4);
            memset(gb1, 0, B1_SZ*4); memset(gb2, 0, B2_SZ*4); memset(gb3, 0, B3_SZ*4);
            int acc = 0;

            for (int step = 0; step + BATCH_SIZE <= ns; step += BATCH_SIZE) {
                // Forward L1
                IOSurfaceLock(k1->ioIn, 0, NULL);
                float *inp = IOSurfaceGetBaseAddress(k1->ioIn);
                for (int b = 0; b < BATCH_SIZE; b++) {
                    float *s = S + idx[step+b]*(nf+1);
                    for (int f = 0; f < IN_DIM; f++) inp[f*BATCH_SIZE+b] = s[f];
                }
                IOSurfaceUnlock(k1->ioIn, 0, NULL);
                eval(k1);

                IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *o1 = IOSurfaceGetBaseAddress(k1->ioOut);
                for (int j = 0; j < H1; j++)
                    for (int b = 0; b < BATCH_SIZE; b++) {
                        float v = o1[j*BATCH_SIZE+b] + b1[j];
                        a1[j*BATCH_SIZE+b] = v;
                        r1[j*BATCH_SIZE+b] = v > 0 ? v : 0;
                    }
                IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Forward L2
                IOSurfaceLock(k2->ioIn, 0, NULL);
                memcpy(IOSurfaceGetBaseAddress(k2->ioIn), r1, H1*BATCH_SIZE*4);
                IOSurfaceUnlock(k2->ioIn, 0, NULL);
                eval(k2);

                IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *o2 = IOSurfaceGetBaseAddress(k2->ioOut);
                for (int j = 0; j < H2; j++)
                    for (int b = 0; b < BATCH_SIZE; b++) {
                        float v = o2[j*BATCH_SIZE+b] + b2[j];
                        a2[j*BATCH_SIZE+b] = v;
                        r2[j*BATCH_SIZE+b] = v > 0 ? v : 0;
                    }
                IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Forward L3
                IOSurfaceLock(k3->ioIn, 0, NULL);
                memcpy(IOSurfaceGetBaseAddress(k3->ioIn), r2, H2*BATCH_SIZE*4);
                IOSurfaceUnlock(k3->ioIn, 0, NULL);
                eval(k3);

                IOSurfaceLock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *o3 = IOSurfaceGetBaseAddress(k3->ioOut);

                float d3[BATCH_SIZE];
                for (int b = 0; b < BATCH_SIZE; b++) {
                    float logit = o3[b] + b3[0];
                    float pred = 1.0f/(1.0f+expf(-logit));
                    out[b] = pred;
                    float raw_label = S[idx[step+b]*(nf+1)+nf];
                    // Label smoothing
                    float label = raw_label * (1 - LABEL_SMOOTH) + 0.5f * LABEL_SMOOTH;
                    float w = raw_label > 0.5f ? wp : wn;
                    float p = fmaxf(fminf(pred, 1-1e-7f), 1e-7f);
                    eloss += w * -(label*logf(p) + (1-label)*logf(1-p));
                    int pc = pred > 0.5f, tc = raw_label > 0.5f;
                    if (pc&&tc) tp++; else if (!pc&&!tc) tn++;
                    else if (pc&&!tc) fp++; else fn++;
                    d3[b] = clip(w * (pred - label));
                }
                IOSurfaceUnlock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Backward L3→L2
                for (int b = 0; b < BATCH_SIZE; b++) {
                    gb3[0] += d3[b];
                    for (int j = 0; j < H2; j++)
                        gW3[j] += d3[b] * r2[j*BATCH_SIZE+b];
                }
                float d2[H2*BATCH_SIZE];
                for (int b = 0; b < BATCH_SIZE; b++)
                    for (int j = 0; j < H2; j++) {
                        float dr = W3[j] * d3[b];
                        d2[j*BATCH_SIZE+b] = a2[j*BATCH_SIZE+b] > 0 ? dr : 0;
                    }

                // Backward L2→L1
                for (int b = 0; b < BATCH_SIZE; b++)
                    for (int j = 0; j < H2; j++) {
                        gb2[j] += d2[j*BATCH_SIZE+b];
                        for (int i = 0; i < H1; i++)
                            gW2[j*H1+i] += d2[j*BATCH_SIZE+b] * r1[i*BATCH_SIZE+b];
                    }
                float d1[H1*BATCH_SIZE];
                memset(d1, 0, sizeof(d1));
                for (int b = 0; b < BATCH_SIZE; b++)
                    for (int i = 0; i < H1; i++) {
                        float sum = 0;
                        for (int j = 0; j < H2; j++) sum += W2[j*H1+i] * d2[j*BATCH_SIZE+b];
                        d1[i*BATCH_SIZE+b] = a1[i*BATCH_SIZE+b] > 0 ? sum : 0;
                    }

                // Backward L1→input
                for (int b = 0; b < BATCH_SIZE; b++) {
                    float *x = S + idx[step+b]*(nf+1);
                    for (int i = 0; i < H1; i++) {
                        gb1[i] += d1[i*BATCH_SIZE+b];
                        for (int f = 0; f < IN_DIM; f++)
                            gW1[i*IN_DIM+f] += d1[i*BATCH_SIZE+b] * x[f];
                    }
                }

                acc++;
                if (acc >= ACCUM_STEPS) {
                    at++;
                    float sc = 1.0f/(acc*BATCH_SIZE);
                    #define ADAM(w,g,m,v,sz) do { for (int _i=0;_i<(sz);_i++) { \
                        float gi = clip((g)[_i]*sc); \
                        (m)[_i] = ba*(m)[_i]+(1-ba)*gi; \
                        (v)[_i] = bb*(v)[_i]+(1-bb)*gi*gi; \
                        float mh = (m)[_i]/(1-powf(ba,at)); \
                        float vh = (v)[_i]/(1-powf(bb,at)); \
                        (w)[_i] -= lr*mh/(sqrtf(vh)+eps); \
                    } } while(0)
                    ADAM(W1,gW1,mW1,vW1,W1_SZ); ADAM(W2,gW2,mW2,vW2,W2_SZ);
                    ADAM(W3,gW3,mW3,vW3,W3_SZ);
                    ADAM(b1,gb1,mb1,vb1,B1_SZ); ADAM(b2,gb2,mb2,vb2,B2_SZ);
                    ADAM(b3,gb3,mb3,vb3,B3_SZ);
                    #undef ADAM
                    memset(gW1,0,W1_SZ*4); memset(gW2,0,W2_SZ*4); memset(gW3,0,W3_SZ*4);
                    memset(gb1,0,B1_SZ*4); memset(gb2,0,B2_SZ*4); memset(gb3,0,B3_SZ*4);
                    acc = 0;
                    if (g_cc + 3 <= MAX_COMPILES) {
                        freeK(k1); freeK(k2); freeK(k3);
                        k1 = comp(IN_DIM, H1, BATCH_SIZE, W1);
                        k2 = comp(H1, H2, BATCH_SIZE, W2);
                        k3 = comp(H2, OUT_DIM, BATCH_SIZE, W3);
                    }
                }
            }
            free(idx);
            double ms = tb_ms(mach_absolute_time()-t0);
            int tot = tp+tn+fp+fn;
            printf("E%2d: loss=%.4f acc=%.1f%% P=%.1f%% R=%.1f%% negAcc=%.1f%% lr=%.5f  %.0fms c=%d\n",
                ep, eloss/tot, 100.0f*(tp+tn)/tot,
                tp>0?100.0f*tp/(tp+fp):0, tp>0?100.0f*tp/(tp+fn):0,
                (tn+fp)>0?100.0f*tn/(tn+fp):0, lr, ms, g_cc);

            if (g_cc + 36 > MAX_COMPILES) {
                // Save checkpoint
                FILE *f = fopen(CKPT_PATH, "wb");
                CkptHeader ch = {CKPT_MAGIC, ep+1, at, 0};
                fwrite(&ch, sizeof(ch), 1, f);
                fwrite(W1, 4, W1_SZ, f); fwrite(W2, 4, W2_SZ, f); fwrite(W3, 4, W3_SZ, f);
                fwrite(b1, 4, B1_SZ, f); fwrite(b2, 4, B2_SZ, f); fwrite(b3, 4, B3_SZ, f);
                fwrite(mW1, 4, W1_SZ, f); fwrite(vW1, 4, W1_SZ, f);
                fwrite(mW2, 4, W2_SZ, f); fwrite(vW2, 4, W2_SZ, f);
                fwrite(mW3, 4, W3_SZ, f); fwrite(vW3, 4, W3_SZ, f);
                fwrite(mb1, 4, B1_SZ, f); fwrite(vb1, 4, B1_SZ, f);
                fwrite(mb2, 4, B2_SZ, f); fwrite(vb2, 4, B2_SZ, f);
                fwrite(mb3, 4, B3_SZ, f); fwrite(vb3, 4, B3_SZ, f);
                fclose(f);
                printf("[exec() restart after epoch %d, c=%d]\n", ep, g_cc);
                freeK(k1); freeK(k2); freeK(k3);
                execl(argv[0], argv[0], dp, "--resume", NULL);
                perror("execl"); return 1;
            }
        }
        freeK(k1); freeK(k2); freeK(k3);
        }

eval_only:;
        printf("\n=== Final Evaluation ===\n");
        K *ek1 = comp(IN_DIM, H1, BATCH_SIZE, W1);
        K *ek2 = comp(H1, H2, BATCH_SIZE, W2);
        K *ek3 = comp(H2, OUT_DIM, BATCH_SIZE, W3);
        if (!ek1||!ek2||!ek3) { printf("Eval compile fail\n"); return 1; }

        int tp2=0,tn2=0,fp2=0,fn2=0;
        for (int i = 0; i+BATCH_SIZE <= ns; i += BATCH_SIZE) {
            IOSurfaceLock(ek1->ioIn, 0, NULL);
            float *inp = IOSurfaceGetBaseAddress(ek1->ioIn);
            for (int b = 0; b < BATCH_SIZE; b++) {
                float *s = S+(i+b)*(nf+1);
                for (int f = 0; f < IN_DIM; f++) inp[f*BATCH_SIZE+b] = s[f];
            }
            IOSurfaceUnlock(ek1->ioIn, 0, NULL);
            eval(ek1);

            IOSurfaceLock(ek1->ioOut, kIOSurfaceLockReadOnly, NULL);
            float *o1 = IOSurfaceGetBaseAddress(ek1->ioOut);
            IOSurfaceLock(ek2->ioIn, 0, NULL);
            float *i2 = IOSurfaceGetBaseAddress(ek2->ioIn);
            for (int j = 0; j < H1*BATCH_SIZE; j++) i2[j] = fmaxf(o1[j]+b1[j/BATCH_SIZE], 0);
            IOSurfaceUnlock(ek1->ioOut, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(ek2->ioIn, 0, NULL);
            eval(ek2);

            IOSurfaceLock(ek2->ioOut, kIOSurfaceLockReadOnly, NULL);
            float *o2 = IOSurfaceGetBaseAddress(ek2->ioOut);
            IOSurfaceLock(ek3->ioIn, 0, NULL);
            float *i3 = IOSurfaceGetBaseAddress(ek3->ioIn);
            for (int j = 0; j < H2*BATCH_SIZE; j++) i3[j] = fmaxf(o2[j]+b2[j/BATCH_SIZE], 0);
            IOSurfaceUnlock(ek2->ioOut, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(ek3->ioIn, 0, NULL);
            eval(ek3);

            IOSurfaceLock(ek3->ioOut, kIOSurfaceLockReadOnly, NULL);
            float *o3 = IOSurfaceGetBaseAddress(ek3->ioOut);
            for (int b = 0; b < BATCH_SIZE; b++) {
                float pred = 1.0f/(1.0f+expf(-(o3[b]+b3[0])));
                float label = S[(i+b)*(nf+1)+nf];
                int pc = pred > 0.5f, tc = label > 0.5f;
                if (pc&&tc) tp2++; else if (!pc&&!tc) tn2++;
                else if (pc&&!tc) fp2++; else fn2++;
            }
            IOSurfaceUnlock(ek3->ioOut, kIOSurfaceLockReadOnly, NULL);
        }
        int tot2 = tp2+tn2+fp2+fn2;
        printf("Accuracy: %.1f%% (%d/%d)\n", 100.0f*(tp2+tn2)/tot2, tp2+tn2, tot2);
        printf("Precision: %.1f%%  Recall: %.1f%%\n",
            tp2>0?100.0f*tp2/(tp2+fp2):0, tp2>0?100.0f*tp2/(tp2+fn2):0);
        printf("Neg accuracy: %.1f%% (TN=%d FP=%d)\n",
            (tn2+fp2)>0?100.0f*tn2/(tn2+fp2):0, tn2, fp2);
        printf("TP=%d TN=%d FP=%d FN=%d\n", tp2, tn2, fp2, fn2);

        printf("\nInference benchmark (1000 passes, 3-layer)...\n");
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 1000; i++) { eval(ek1); eval(ek2); eval(ek3); }
        double ms = tb_ms(mach_absolute_time()-t0);
        printf("Total: %.1fms  Per-inference: %.3fms (%.0f/sec)\n", ms, ms/1000.0, 1e6/ms);

        // Save model
        FILE *mf = fopen("btc_mlp3_final.bin", "wb");
        if (mf) {
            int magic = 0x4D4C3346;
            fwrite(&magic, 4, 1, mf);
            int dims[] = {IN_DIM, H1, H2, OUT_DIM};
            fwrite(dims, 4, 4, mf);
            fwrite(W1, 4, W1_SZ, mf); fwrite(W2, 4, W2_SZ, mf); fwrite(W3, 4, W3_SZ, mf);
            fwrite(b1, 4, B1_SZ, mf); fwrite(b2, 4, B2_SZ, mf); fwrite(b3, 4, B3_SZ, mf);
            fclose(mf);
            printf("\nModel saved to btc_mlp3_final.bin\n");
        }

        freeK(ek1); freeK(ek2); freeK(ek3);
        printf("Done.\n");
    }
    return 0;
}