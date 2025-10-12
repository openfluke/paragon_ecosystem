#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "paragon.h"

static void* must_dlsym_try(void* so, const char* name){
  if(!so) return NULL;
  dlerror();
  void* p = dlsym(so, name);
  (void)dlerror();
  return p;
}

static void* resolve_any(void* so, const char* const names[]){
  for(int i=0; names[i]; ++i){
    void* p = must_dlsym_try(so, names[i]);
    if(p) return p;
  }
  return NULL;
}

int paragon_load(ParagonAPI* api, const char* so_path){
  memset(api, 0, sizeof(*api));
  api->so = dlopen(so_path && *so_path ? so_path : NULL, RTLD_NOW | RTLD_GLOBAL);
  if(!api->so){
    fprintf(stderr, "dlopen failed (%s): %s\n", so_path?so_path:"<NULL>", dlerror());
    return 1; /* keep running; we’ll no-op gracefully */
  }

  /* Try all reasonable name variants seen in the wild */
  const char* NEW5[] = {
    "Paragon_NewNetworkFloat32",
    "Teleport_NewNetworkFloat32",
    "NewNetworkFloat32",
    NULL
  };
  const char* NEW3[] = {
    "Paragon_NewNetworkFloat32_JSON",
    "Teleport_NewNetworkFloat32_JSON",
    "NewNetworkFloat32_JSON",
    NULL
  };
  const char* CALLN[] = {
    "Paragon_Call",
    "Teleport_Call",
    "Call",
    NULL
  };

  api->New5 = (fn_NewNetworkFloat32_5) resolve_any(api->so, NEW5);
  api->New3 = (fn_NewNetworkFloat32_3) resolve_any(api->so, NEW3);
  api->Call = (fn_Call)                 resolve_any(api->so, CALLN);

  if(!api->New5 && !api->New3 && !api->Call){
    fprintf(stderr, "No compatible symbols found: NewNetworkFloat32/Call.\n");
  }
  return 1;
}

void paragon_unload(ParagonAPI* api){
  if(!api) return;
  if(api->so){ dlclose(api->so); api->so = NULL; }
  api->New5 = NULL; api->New3 = NULL; api->Call = NULL;
}

/* Robust handle parser (bare integer string, {handle: N}, {result:{handle:N}}, …) */
ParagonHandle paragon_parse_handle(const char* txt){
  if(!txt) return -1;
  /* 1) bare integer */
  char* endp = NULL;
  long long v = strtoll(txt, &endp, 10);
  if(endp && *endp=='\0') return v;

  const char* keys[] = {
    "\"handle\"", "\"Handle\"", "\"id\"", "\"ID\"",
    "\"network_handle\"", "\"NetworkHandle\"",
    "\"h\"", "\"H\"",
    NULL
  };
  for(int pass=0; pass<2; ++pass){
    const char* hay = txt;
    if(pass==1){
      const char* r = strstr(txt, "\"result\"");
      if(!r) break;
      hay = r;
    }
    for(int i=0; keys[i]; ++i){
      const char* p = strstr(hay, keys[i]);
      if(!p) continue;
      p = strchr(p, ':'); if(!p) continue;
      while(*p==':' || *p==' ' || *p=='\t' || *p=='\"') ++p;
      char* e=NULL; long long n = strtoll(p, &e, 10);
      if(e!=p) return n;
    }
  }
  return -1;
}

char* paragon_call0(ParagonAPI* api, ParagonHandle h, const char* method){
  if(!api || !api->Call) return NULL;
  return api->Call(h, method, "[]");
}

/* Create net via any available route:
   - 5-arg NewNetworkFloat32 (preferred)
   - 3-arg NewNetworkFloat32 (fallback)
   - meta: Call(0,"NewNetworkFloat32","[...]") */
char* paragon_new_net_any(ParagonAPI* api,
                          const char* layers_json,
                          const char* activs_json,
                          const char* trainable_json,
                          bool prefer_gpu,
                          bool expose_methods_json)
{
  if(api->New5){
    return api->New5(layers_json, activs_json, trainable_json, prefer_gpu, expose_methods_json);
  }
  if(api->New3){
    return api->New3(layers_json, activs_json, trainable_json);
  }
  if(api->Call){
    /* Build args: [layers, activs, trainable, prefer_gpu, expose] */
    char* args = NULL;
    size_t n = strlen(layers_json)+strlen(activs_json)+strlen(trainable_json)+64;
    args = (char*)malloc(n);
    snprintf(args, n, "[%s,%s,%s,%s,%s]",
             layers_json, activs_json, trainable_json,
             prefer_gpu ? "true":"false",
             expose_methods_json ? "true":"false");
    char* r = api->Call(0, "NewNetworkFloat32", args);
    free(args);
    return r;
  }
  return NULL;
}
