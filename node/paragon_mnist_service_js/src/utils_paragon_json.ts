// utils_paragon_json.ts
import { promises as fs } from "node:fs";

/** ───────────────────────── Schema (matches Go exactly) ───────────────────── */

export type SConn = {
  layer: number; // source layer index (L)
  x: number; // source neuron X
  y: number; // source neuron Y
  w: number; // weight
};

export type SNeuron = {
  b: number; // Bias
  a: string; // Activation
  in: SConn[]; // Inputs
};

export type SLayer = {
  w: number; // width
  h: number; // height
  n: SNeuron[][]; // neurons [row y][col x]

  // optional replay fields
  re_enabled?: boolean;
  re_offset?: number;
  re_phase?: string;
  re_max?: number;
  re_budget?: number;
};

export type SNet = {
  /** "type" in Go: "float32" | "float64" | "int8" ... */
  type: string;
  layers: SLayer[];
};

/** ───────────────────────── Helpers: validation ───────────────────────────── */

function assert(condition: any, msg: string): asserts condition {
  if (!condition) throw new Error(`ParagonJSON: ${msg}`);
}

export function validateSNet(s: SNet): void {
  assert(typeof s === "object" && s !== null, "sNet must be an object");
  assert(typeof s.type === "string" && s.type.length > 0, "missing type");
  assert(
    Array.isArray(s.layers) && s.layers.length > 0,
    "layers must be non-empty array"
  );

  s.layers.forEach((L, li) => {
    assert(Number.isInteger(L.w) && L.w > 0, `layer ${li}: invalid width`);
    assert(Number.isInteger(L.h) && L.h > 0, `layer ${li}: invalid height`);
    assert(
      Array.isArray(L.n) && L.n.length === L.h,
      `layer ${li}: n rows != h`
    );

    for (let y = 0; y < L.h; y++) {
      const row = L.n[y];
      assert(
        Array.isArray(row) && row.length === L.w,
        `layer ${li} row ${y}: width mismatch`
      );
      for (let x = 0; x < L.w; x++) {
        const nn = row[x];
        assert(
          typeof nn.b === "number",
          `layer ${li} (${x},${y}): missing bias b`
        );
        assert(
          typeof nn.a === "string",
          `layer ${li} (${x},${y}): missing activation a`
        );
        assert(
          Array.isArray(nn.in),
          `layer ${li} (${x},${y}): in must be array`
        );
        nn.in.forEach((c, k) => {
          assert(
            Number.isInteger(c.layer),
            `conn ${li}/${y}/${x}#${k}: layer must be int`
          );
          assert(
            Number.isInteger(c.x),
            `conn ${li}/${y}/${x}#${k}: x must be int`
          );
          assert(
            Number.isInteger(c.y),
            `conn ${li}/${y}/${x}#${k}: y must be int`
          );
          assert(
            typeof c.w === "number",
            `conn ${li}/${y}/${x}#${k}: w must be number`
          );
        });
      }
    }
  });
}

/** ─────────────── In-memory convertors (your JS runtime shape ↔ sNet) ───────
 * If you already have a JS "Network" object, map its fields here.
 * These are no-ops placeholders until you wire them to your runtime.
 */

export type JsConnection = { L: number; X: number; Y: number; W: number };
export type JsNeuron = {
  Bias: number;
  Activation: string;
  Inputs: JsConnection[];
};
export type JsLayer = {
  Width: number;
  Height: number;
  Neurons: JsNeuron[][];
  ReplayEnabled?: boolean;
  ReplayOffset?: number;
  ReplayPhase?: string;
  MaxReplay?: number;
  ReplayBudget?: number;
};
export type JsNetwork = {
  TypeName: string; // "float32", "float64", ...
  Layers: JsLayer[];
  OutputLayer?: number;
};

export function toSerializable(net: JsNetwork): SNet {
  const s: SNet = {
    type: net.TypeName ?? "float32",
    layers: net.Layers.map((L) => ({
      w: L.Width,
      h: L.Height,
      n: L.Neurons.map((row) =>
        row.map((src) => ({
          b: src.Bias,
          a: src.Activation,
          in: src.Inputs.map((c) => ({
            layer: c.L,
            x: c.X,
            y: c.Y,
            w: c.W,
          })),
        }))
      ),
      re_enabled: L.ReplayEnabled || undefined,
      re_offset: L.ReplayOffset || undefined,
      re_phase: L.ReplayPhase || undefined,
      re_max: L.MaxReplay || undefined,
      re_budget: L.ReplayBudget || undefined,
    })),
  };
  validateSNet(s);
  return s;
}

export function fromSerializable(s: SNet): JsNetwork {
  validateSNet(s);
  const js: JsNetwork = {
    TypeName: s.type,
    Layers: s.layers.map((sl) => ({
      Width: sl.w,
      Height: sl.h,
      Neurons: sl.n.map((row) =>
        row.map((sn) => ({
          Bias: sn.b,
          Activation: sn.a,
          Inputs: sn.in.map((c) => ({
            L: c.layer,
            X: c.x,
            Y: c.y,
            W: c.w,
          })),
        }))
      ),
      ReplayEnabled: sl.re_enabled,
      ReplayOffset: sl.re_offset,
      ReplayPhase: sl.re_phase,
      MaxReplay: sl.re_max,
      ReplayBudget: sl.re_budget,
    })),
    // OutputLayer is derivable the same way Go sets it:
    OutputLayer: s.layers.length ? s.layers.length - 1 : undefined,
  };
  return js;
}

/** ─────────────────────── Disk I/O helpers (Node) ─────────────────────────── */

export async function saveNetworkJSON(
  net: JsNetwork,
  path: string
): Promise<void> {
  const s = toSerializable(net);
  const pretty = JSON.stringify(s, null, " ");
  await fs.writeFile(path, pretty, { mode: 0o644 });
}

export async function loadNetworkJSON(path: string): Promise<SNet> {
  const data = await fs.readFile(path, "utf8");
  const s: SNet = JSON.parse(data);
  validateSNet(s);
  return s;
}

/** Handy one-liners if you don’t keep a JS runtime shape at all: */
export async function saveSerializableJSON(
  s: SNet,
  path: string
): Promise<void> {
  validateSNet(s);
  await fs.writeFile(path, JSON.stringify(s, null, " "), { mode: 0o644 });
}
export async function loadAsJsNetwork(path: string): Promise<JsNetwork> {
  const s = await loadNetworkJSON(path);
  return fromSerializable(s);
}
