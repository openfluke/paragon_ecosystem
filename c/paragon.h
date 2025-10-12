#ifndef PARAGON_H
#define PARAGON_H

#include <stdbool.h>

typedef long long ParagonHandle;

/* Candidate types the .so might expose */
typedef char* (*fn_NewNetworkFloat32_5)(
  const char* layers_json,
  const char* activs_json,
  const char* trainable_json,
  bool use_gpu,
  bool expose_methods_json
);

typedef char* (*fn_NewNetworkFloat32_3)(
  const char* layers_json,
  const char* activs_json,
  const char* trainable_json
);

/* Common Call signature we’ve observed */
typedef char* (*fn_Call)(
  ParagonHandle handle,
  const char* method_utf8,
  const char* args_json_utf8
);

typedef struct {
  void* so;
  fn_NewNetworkFloat32_5 New5;
  fn_NewNetworkFloat32_3 New3;
  fn_Call                 Call;
} ParagonAPI;

int  paragon_load(ParagonAPI* api, const char* so_path);   /* 1 = ok */
void paragon_unload(ParagonAPI* api);

ParagonHandle paragon_parse_handle(const char* txt);       /* -1 on failure */
char*         paragon_call0(ParagonAPI* api, ParagonHandle h, const char* method);

/* High-level helpers matching your C# flow */
char* paragon_new_net_any(ParagonAPI* api,
                          const char* layers_json,
                          const char* activs_json,
                          const char* trainable_json,
                          bool prefer_gpu,
                          bool expose_methods_json);

#endif
