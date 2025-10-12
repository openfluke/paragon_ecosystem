#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "paragon.h"
#include <stdarg.h> 

static double now_ms(){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

static void append(char** buf, size_t* cap, size_t* len, const char* fmt, ...){
  va_list ap; va_start(ap, fmt);
  char tmp[512]; int w = vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  if(w<0) exit(1);
  if(*len + w + 1 > *cap){
    *cap = (*cap + w + 1024) * 2;
    *buf = (char*)realloc(*buf, *cap);
    if(!*buf) exit(1);
  }
  memcpy(*buf + *len, tmp, w); *len += w; (*buf)[*len]=0;
}

static char* json_layers(const int* dims, int ndims){
  size_t cap=1024,len=0; char* b=malloc(cap); if(!b) exit(1); b[0]=0;
  append(&b,&cap,&len,"[");
  for(int i=0;i<ndims;i++){
    if(i) append(&b,&cap,&len,",");
    append(&b,&cap,&len,"{\"Width\":%d,\"Height\":1}", dims[i]);
  }
  append(&b,&cap,&len,"]");
  return b;
}
static char* json_activs(int ndims){
  size_t cap=512,len=0; char* b=malloc(cap); if(!b) exit(1); b[0]=0;
  append(&b,&cap,&len,"[\"linear\"");
  for(int i=1;i<ndims-1;i++) append(&b,&cap,&len,",\"relu\"");
  append(&b,&cap,&len,",\"softmax\"]");
  return b;
}
static char* json_trainable(int ndims){
  size_t cap=256,len=0; char* b=malloc(cap); if(!b) exit(1); b[0]=0;
  append(&b,&cap,&len,"[");
  for(int i=0;i<ndims;i++) append(&b,&cap,&len, i?",true":"true");
  append(&b,&cap,&len,"]");
  return b;
}
static char* json_fixed784(){
  /* Reproduce C#’s vector: LCG 1664525/1013904223, /0xffffffff, rounded-ish. */
  size_t cap=20000,len=0; char* b=malloc(cap); if(!b) exit(1); b[0]=0;
  append(&b,&cap,&len,"[[");
  unsigned s = 123u;
  for(int i=0;i<784;i++){
    s = s*1664525u + 1013904223u;
    double v = (double)s / 4294967295.0;
    append(&b,&cap,&len, (i?",%.6f":"%.6f"), v);
  }
  append(&b,&cap,&len,"]]");
  return b;
}

static int parse_vector_tolerant(const char* txt, double* out, int maxn){
  if(!txt) return 0;
  const char* p = strchr(txt, '[');
  const char* q = strrchr(txt, ']');
  if(!p || !q || q<=p) return 0;
  int n=0;
  for(const char* s=p+1; s<q && n<maxn; ){
    char* e=NULL; double v = strtod(s,&e);
    if(e==s){ ++s; continue; }
    out[n++] = v; s = e;
  }
  return n;
}

static void try_gpu_enable(ParagonAPI* api, ParagonHandle h){
  if(!api || !api->Call) return;
  /* best-effort knobs, all optional */
  paragon_call0(api,h,"SetWebGPUNative");    /* some builds use bool default=true */
  paragon_call0(api,h,"WebGPUNativeOn");
  api->Call(h,"Configure","[{\"WebGPUNative\":true}]");
  api->Call(h,"SetOptions","[{\"WebGPUNative\":true}]");
  api->Call(h,"SetField","[\"WebGPUNative\",true]");
  /* generic message-based */
  api->Call(h,"Call","[\"SetWebGPUNative\",[true]]");
}

static void run_one(ParagonAPI* api, const char* id, const int* dims, int ndims){
  char* layers = json_layers(dims, ndims);
  char* activs = json_activs(ndims);
  char* fully  = json_trainable(ndims);
  char* xin    = json_fixed784();

  printf("\n=== %s (%d", id, dims[0]);
  for(int i=1;i<ndims;i++) printf("→%d", dims[i]);
  printf(") ===\n");

  char* newr = paragon_new_net_any(api, layers, activs, fully, false, false);
  ParagonHandle h = paragon_parse_handle(newr);
  if(h<=0){
    fprintf(stderr, "NewNetwork failed or missing. newr=%s\n", newr?newr:"<null>");
    goto out;
  }

  /* GPU init */
  double t_gpu_init_s = now_ms();
  char* adapter = paragon_call0(api,h,"InitializeOptimizedGPU");
  double t_gpu_init_e = now_ms();
  (void)adapter; /* we print via raw below */
  try_gpu_enable(api,h); /* optional paths */

  /* CPU pass */
  double t0 = now_ms();
  {
    size_t n=strlen(xin)+3; char* args=malloc(n); snprintf(args,n,"[%s]", xin);
    (void)api->Call(h,"Forward",args); free(args);
  }
  char* outA = paragon_call0(api,h,"ExtractOutput");
  double t1 = now_ms();

  /* GPU pass */
  (void) paragon_call0(api,h,"ToggleGPU");  /* optional; ignored if missing */
  double tg0 = now_ms();
  {
    size_t n=strlen(xin)+3; char* args=malloc(n); snprintf(args,n,"[%s]", xin);
    (void)api->Call(h,"Forward",args); free(args);
  }
  char* outB = paragon_call0(api,h,"ExtractOutput");
  double tg1 = now_ms();

  double a[1024]={0}, b[1024]={0};
  int na = parse_vector_tolerant(outA, a, 1024);
  int nb = parse_vector_tolerant(outB, b, 1024);
  int n  = na<nb?na:nb;
  double mae=0.0, mx=0.0;
  for(int i=0;i<n;i++){ double d=fabs(a[i]-b[i]); mae+=d; if(d>mx) mx=d; }
  if(n>0) mae/=n;

  /* Console like C# */
  /* shape/estMB */
  long long params=0;
  for(int i=0;i<ndims-1;i++) params += (long long)dims[i]*dims[i+1];
  for(int i=1;i<ndims;i++)   params += dims[i];
  double estMB = params * 4.0 / (1024.0*1024.0);

  printf("Shape: ");
  for(int i=0;i<ndims;i++){ if(i) printf(" → "); printf("%d", dims[i]); }
  printf("   (~weights %.2f MB)\n", estMB);

  printf("GPU init: %s  in %.2f ms\n",
    adapter && *adapter ? adapter : "{}", (t_gpu_init_e - t_gpu_init_s));

  double cpu_ms = (t1 - t0);
  double gpu_ms = (tg1 - tg0);
  printf("CPU  ⏱ %.3f ms\n", cpu_ms);
  printf("GPU  ⏱ %.3f ms\n", gpu_ms);
  printf("Speedup: %.2fx\n", (gpu_ms>0? cpu_ms/gpu_ms : 0.0));
  printf("Δ(CPU vs GPU)  mae=%0.00E  max=%0.00E\n", mae, mx);
  printf("CPU ExtractOutput: %s\n", outA?outA:"");
  printf("GPU ExtractOutput: %s\n", outB?outB:"");

out:
  free(layers); free(activs); free(fully); free(xin);
}

int main(int argc, char** argv){
  const char* so = (argc>1 && argv[1][0]!='-') ? argv[1] : NULL;

  ParagonAPI api;
  (void)paragon_load(&api, so);

  const int S1[]  = {784,  64, 10};
  const int S2[]  = {784, 128, 10};
  const int S3[]  = {784, 256, 10};
  const int M1[]  = {784, 256, 256, 10};
  const int M2[]  = {784, 384, 384, 10};
  const int M3[]  = {784, 512, 512, 10};
  const int L1[]  = {784, 768, 768, 768, 10};
  const int L2[]  = {784,1024,1024,1024, 10};
  const int XL1[] = {784,1536,1536,1536,1536,10};
  const int XL2[] = {784,2048,2048,2048,2048,10};

  run_one(&api,"S1", S1,  (int)(sizeof(S1)/sizeof(S1[0])));
  run_one(&api,"S2", S2,  (int)(sizeof(S2)/sizeof(S2[0])));
  run_one(&api,"S3", S3,  (int)(sizeof(S3)/sizeof(S3[0])));
  run_one(&api,"M1", M1,  (int)(sizeof(M1)/sizeof(M1[0])));
  run_one(&api,"M2", M2,  (int)(sizeof(M2)/sizeof(M2[0])));
  run_one(&api,"M3", M3,  (int)(sizeof(M3)/sizeof(M3[0])));
  run_one(&api,"L1", L1,  (int)(sizeof(L1)/sizeof(L1[0])));
  run_one(&api,"L2", L2,  (int)(sizeof(L2)/sizeof(L2[0])));
  run_one(&api,"XL1",XL1, (int)(sizeof(XL1)/sizeof(XL1[0])));
  run_one(&api,"XL2",XL2, (int)(sizeof(XL2)/sizeof(XL2[0])));

  paragon_unload(&api);
  return 0;
}
