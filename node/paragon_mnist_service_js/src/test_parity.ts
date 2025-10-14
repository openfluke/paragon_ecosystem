import { writeFileSync } from "node:fs";
import { fetch } from "undici";

const BASE = process.env.BASE ?? "http://127.0.0.1:8000";

function drift(a: number[], b: number[]) {
  const diffs = a.map((v, i) => Math.abs(v - b[i]));
  const maxAbs = Math.max(...diffs);
  const meanAbs = diffs.reduce((x, y) => x + y, 0) / diffs.length;
  const l2 = Math.sqrt(diffs.reduce((x, y) => x + y * y, 0));
  return { max_abs_diff: maxAbs, mean_abs_diff: meanAbs, l2_diff: l2 };
}

(async () => {
  const health = await (await fetch(`${BASE}/health`)).json();
  console.log("health:", health);

  const list = await (await fetch(`${BASE}/images/list`)).json();
  console.log("images:", list.images);

  const parity = await (await fetch(`${BASE}/parity`)).json();
  console.log(`mismatches: ${parity.mismatches}/${parity.total}`);

  // Compute drift table (when GPU available)
  const rows = [
    [
      "image",
      "pred_cpu",
      "pred_gpu",
      "max_abs_diff",
      "mean_abs_diff",
      "l2_diff",
    ],
  ];
  for (const r of parity.results) {
    if (!r.gpu) {
      rows.push([r.image, r.cpu?.pred ?? "", "", "", "", ""]);
      continue;
    }
    const d = drift(r.cpu.probs, r.gpu.probs);
    rows.push([
      r.image,
      r.cpu.pred,
      r.gpu.pred,
      d.max_abs_diff,
      d.mean_abs_diff,
      d.l2_diff,
    ]);
  }

  writeFileSync("parity_report.json", JSON.stringify(parity, null, 2));
  writeFileSync(
    "parity_summary.csv",
    rows.map((row) => row.join(",")).join("\n")
  );
  console.log("🎯 Wrote parity_report.json & parity_summary.csv");
})();
