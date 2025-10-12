import { BenchSuite, PRESETS } from "./bench";

const main = async () => {
  const suite = new BenchSuite();
  // suite.enableCsv("bench_results.csv"); // optional CSV
  await suite.runAll(PRESETS.MNIST_ZOO);
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
