// bun run compare.ts
import { initPortal } from "@openfluke/portal";

type Cfg = {
  name: string;
  layers: { Width: number; Height: number }[];
  activations: string[];
};

const configs: Cfg[] = [
  {
    name: "Alpha",
    layers: [
      { Width: 1, Height: 1 },
      { Width: 2, Height: 1 },
      { Width: 3, Height: 1 },
    ],
    activations: ["linear", "relu", "softmax"],
  },
  {
    name: "Beta",
    layers: [
      { Width: 1, Height: 1 },
      { Width: 4, Height: 1 },
      { Width: 3, Height: 1 },
    ],
    activations: ["linear", "tanh", "softmax"],
  },
  {
    name: "Gamma",
    layers: [
      { Width: 1, Height: 1 },
      { Width: 3, Height: 1 },
      { Width: 3, Height: 1 },
    ],
    activations: ["linear", "sigmoid", "softmax"],
  },
];

// deterministic "fixed" input — same for CPU & GPU passes
const FIXED_INPUT = JSON.stringify([[[0.42]]]); // shape [1,1,1]

function parseOut(jsonStr: string): number[] {
  // outputs are like "[[...]]" -> flatten
  const a = JSON.parse(jsonStr);
  return Array.isArray(a) ? a.flat(2) : [];
}

function diffs(a: number[], b: number[]) {
  let mae = 0,
    maxd = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i]);
    mae += d;
    if (d > maxd) maxd = d;
  }
  mae /= Math.min(a.length, b.length) || 1;
  return { mae, maxd };
}

async function runOne(portal: any, cfg: Cfg) {
  const nn = portal.NewNetworkFloat32(
    JSON.stringify(cfg.layers),
    JSON.stringify(cfg.activations),
    JSON.stringify(new Array(cfg.layers.length).fill(true))
  );

  // Freeze weights deterministically: [amplitude, seed]
  nn.PerturbWeights(JSON.stringify([0.1, 123])); // constant seed

  // WARMUP (CPU)
  nn.Forward(FIXED_INPUT);
  nn.ExtractOutput();

  // CPU timing
  let t0 = performance.now();
  nn.Forward(FIXED_INPUT);
  let cpuOut = parseOut(nn.ExtractOutput());
  const cpuMs = performance.now() - t0;

  // Try GPU
  let gpuInit = null;
  if (nn.InitializeOptimizedGPU) {
    gpuInit = nn.InitializeOptimizedGPU(); // may log “WebGPU not supported”
  }

  // WARMUP (GPU path)
  nn.Forward(FIXED_INPUT);
  nn.ExtractOutput();

  // GPU timing (same input, same weights)
  t0 = performance.now();
  nn.Forward(FIXED_INPUT);
  const gpuOut = parseOut(nn.ExtractOutput());
  const gpuMs = performance.now() - t0;

  // Diff
  const { mae, maxd } = diffs(cpuOut, gpuOut);

  // Cleanup
  if (nn.CleanupOptimizedGPU) nn.CleanupOptimizedGPU();

  console.log(`\n=== ${cfg.name} ===`);
  console.log(`CPU  ⏱ ${cpuMs.toFixed(3)} ms  out=${JSON.stringify(cpuOut)}`);
  console.log(`GPUi → ${gpuInit}`);
  console.log(`GPU  ⏱ ${gpuMs.toFixed(3)} ms  out=${JSON.stringify(gpuOut)}`);
  console.log(
    `Δ( CPU vs GPU )  mae=${mae.toExponential(2)}  max=${maxd.toExponential(2)}`
  );
  console.log(`Speedup: ${(cpuMs / gpuMs).toFixed(2)}×`);
}

(async () => {
  const portal = await initPortal();
  for (const cfg of configs) {
    await runOne(portal, cfg);
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
