// index.ts — robustly locate @openfluke/portal without depending on dist/loader
import { BenchSuite, PRESETS } from "./bench.ts";

async function resolveInitPortal(): Promise<() => Promise<any>> {
  const tries = [
    "@openfluke/portal",
    "@openfluke/portal/node",
    "@openfluke/portal/dist/index.js",
    "@openfluke/portal/dist/index.mjs",
    "@openfluke/portal/esm/index.js",
    "@openfluke/portal/cjs/index.cjs",
  ];
  let lastErr: any = null;
  for (const id of tries) {
    try {
      const m: any = await import(id);
      if (typeof m.initPortal === "function") return m.initPortal;
    } catch (e) {
      lastErr = e;
    }
  }
  const hint = [
    "Portal couldn’t be imported. Things to check:",
    "• Version installed: npm ls @openfluke/portal",
    "• Which files exist under node_modules/@openfluke/portal/",
    "• If you’re on a monorepo/local build, use a file: path in package.json",
    "• If the package expects Bun-only loader, ask for a Node build or alt entrypoint",
  ].join("\n");
  throw new Error(hint + (lastErr ? `\nLast error: ${lastErr}` : ""));
}

(async () => {
  const initPortal = await resolveInitPortal();
  const suite = new BenchSuite(initPortal);
  // suite.enableCsv("bench_results.csv");
  await suite.runAll(PRESETS.MNIST_ZOO);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
