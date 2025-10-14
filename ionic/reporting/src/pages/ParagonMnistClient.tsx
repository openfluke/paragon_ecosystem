import React, { useEffect, useMemo, useRef, useState } from "react";
import { initPortal } from "@openfluke/portal";

/**
 * Paragon MNIST • Portal (Bulma)
 * Minimal comparison table:
 *   service, digit, p0, p1, ..., p9  (10 d.p.)
 * Rows = services (nodejs, python, golang, wasm/portal)
 * Digit is selected once (top controls) and applied to all services.
 * Per-service Refresh, Copy CSV button.
 */

type Float2D = number[][];
const CLASSES = Array.from({ length: 10 }, (_, i) => i);
const argsJSON = (...a: any[]) => JSON.stringify(a);

function softmax(v: number[]) {
  const m = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - m));
  const Z = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / Z);
}

async function idleFrame() {
  return new Promise<void>((r) => requestAnimationFrame(() => r()));
}

/** Private OOP Portal client (no hooks) */
class MnistClient {
  private portal: any | null = null;
  private NewNetworkFloat32: any | null = null;
  private net: any | null = null;

  async init(): Promise<void> {
    this.portal = await initPortal();
    this.NewNetworkFloat32 = this.portal?.NewNetworkFloat32;
    if (!this.NewNetworkFloat32)
      throw new Error("NewNetworkFloat32 missing from Portal");
  }

  createMnistNet(): void {
    if (!this.NewNetworkFloat32) throw new Error("Call init() first.");
    const L = JSON.stringify([
      { Width: 28, Height: 28 },
      { Width: 256, Height: 1 },
      { Width: 10, Height: 1 },
    ]);
    const A = JSON.stringify(["linear", "relu", "softmax"]);
    const F = JSON.stringify([true, true, true]);
    this.net = this.NewNetworkFloat32(L, A, F);
    if (!this.net) throw new Error("Failed to create network.");
  }

  perturb(scale = 0.05, seed = 1337): void {
    if (!this.net?.PerturbWeights) return;
    this.net.PerturbWeights(argsJSON(scale, seed));
  }

  async loadModelFrom(url: string): Promise<void> {
    if (!this.net) throw new Error("Create the net first.");
    await idleFrame();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} while fetching model`);
    const buf = await resp.arrayBuffer();
    const bytes = Array.from(new Uint8Array(buf).values());
    await idleFrame();
    this.net.UnmarshalJSONModel(argsJSON(bytes));
  }

  run(input28x28: Float2D): number[] {
    if (!this.net) throw new Error("Create the net first.");
    this.net.Forward(argsJSON(input28x28));
    const raw =
      typeof this.net.ExtractOutput === "function"
        ? this.net.ExtractOutput()
        : this.net.GetOutput();
    let probs: number[];
    if (typeof raw === "string") {
      const parsed = JSON.parse(raw);
      probs = Array.isArray(parsed?.[0]) ? parsed[0] : parsed;
    } else if (Array.isArray(raw)) {
      probs = Array.isArray(raw?.[0]) ? raw[0] : raw;
    } else {
      probs = raw;
    }
    return probs;
  }

  async fetchDigitAsInput(
    svcBaseUrl: string,
    digit: number,
    invert = false
  ): Promise<Float2D> {
    const url = `${svcBaseUrl.replace(/\/+$/, "")}/static/images/${digit}.png`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${url}`);
    const blob = await resp.blob();

    const img = new Image();
    const objectUrl = URL.createObjectURL(blob);
    img.src = objectUrl;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Image load failed"));
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D context unavailable");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, 28, 28);
    URL.revokeObjectURL(objectUrl);

    const { data } = ctx.getImageData(0, 0, 28, 28);
    const arr: Float2D = [];
    for (let y = 0; y < 28; y++) {
      const row: number[] = [];
      for (let x = 0; x < 28; x++) {
        const idx = (y * 28 + x) * 4;
        const r = data[idx],
          g = data[idx + 1],
          b = data[idx + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        const v = gray / 255.0;
        row.push(invert ? 1 - v : v);
      }
      arr.push(row);
    }
    return arr;
  }
}

// ---------- Component ----------
type Service = { label: string; url: string; kind: "http" | "portal" };

export default function ParagonMnistPortal(props: {
  modelUrl?: string;
  nodeUrl?: string;
  pythonUrl?: string;
  golangUrl?: string;
}) {
  const {
    modelUrl = "http://127.0.0.1:8001/model",
    nodeUrl = "http://127.0.0.1:8001",
    pythonUrl = "http://127.0.0.1:8002",
    golangUrl = "http://127.0.0.1:8003",
  } = props;

  useEffect(() => {
    document.body.style.overflowY = "auto";
  }, []);

  const services: Service[] = [
    { label: "nodejs", url: nodeUrl, kind: "http" },
    { label: "python", url: pythonUrl, kind: "http" },
    { label: "golang", url: golangUrl, kind: "http" },
    { label: "wasm/portal", url: "browser", kind: "portal" },
  ];

  const pmRef = useRef<MnistClient | null>(null);
  const [ready, setReady] = useState(false);
  const [created, setCreated] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);

  const [digit, setDigit] = useState<number>(9);
  const [invert, setInvert] = useState(false);
  const [clientSoftmax, setClientSoftmax] = useState(false);

  // label -> probs for current digit
  const [rows, setRows] = useState<Record<string, number[]>>({});
  const [loadingService, setLoadingService] = useState<string | null>(null);

  // init portal
  useEffect(() => {
    (async () => {
      try {
        const pm = new MnistClient();
        await pm.init();
        pmRef.current = pm;
        setReady(true);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  const onCreate = () => {
    try {
      pmRef.current!.createMnistNet();
      pmRef.current!.perturb(0.05, 1337);
      setCreated(true);
    } catch (e) {
      console.error(e);
    }
  };

  const onLoad = async () => {
    setLoadingModel(true);
    try {
      await pmRef.current!.loadModelFrom(modelUrl);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingModel(false);
    }
  };

  async function fetchParityDigit(baseUrl: string, d: number) {
    // use /parity then extract that digit's probs to keep consistency
    const resp = await fetch(`${baseUrl.replace(/\/+$/, "")}/parity`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching /parity`);
    const js = await resp.json();
    for (const r of js.results || []) {
      const m = String(r.image || "").match(/(\d+)\.png$/);
      if (!m) continue;
      const idx = Number(m[1]);
      if (idx !== d) continue;
      return (r.cpu?.probs || r.gpu?.probs) as number[] | undefined;
    }
    return undefined;
  }

  async function computePortalDigit(imageSourceUrl: string, d: number) {
    const x = await pmRef.current!.fetchDigitAsInput(imageSourceUrl, d, invert);
    let probs = pmRef.current!.run(x);
    if (clientSoftmax) probs = softmax(probs);
    return probs;
  }

  const refreshService = async (label: string) => {
    setLoadingService(label);
    try {
      const svc = services.find((s) => s.label === label);
      if (!svc) throw new Error(`Unknown service ${label}`);
      let probs: number[] | undefined;
      if (svc.kind === "http") {
        probs = await fetchParityDigit(svc.url, digit);
      } else {
        // portal uses node's images as source of truth
        probs = await computePortalDigit(nodeUrl, digit);
      }
      if (!probs) throw new Error("No probabilities returned");
      setRows((prev) => ({ ...prev, [label]: probs! }));
    } catch (e) {
      console.error(e);
      setRows((prev) => ({ ...prev, [label]: [] }));
    } finally {
      setLoadingService(null);
    }
  };

  const refreshAll = async () => {
    for (const s of services) {
      // sequential to keep UI predictable
      // you can parallelize if you like
      // eslint-disable-next-line no-await-in-loop
      await refreshService(s.label);
    }
  };

  // CSV content for Copy
  const csv = useMemo(() => {
    const header = ["service", "digit", ...CLASSES.map((k) => `p${k}`)];
    const lines = [header.join(",")];
    for (const s of services) {
      const probs = rows[s.label];
      const cols =
        probs && probs.length === 10
          ? probs.map((v) => v.toFixed(10))
          : Array(10).fill("");
      lines.push([s.label, String(digit), ...cols].join(","));
    }
    return lines.join("\n");
  }, [rows, services, digit]);

  const copyCSV = async () => {
    try {
      await navigator.clipboard.writeText(csv);
    } catch {
      // no-op
    }
  };

  const disabled = !ready;
  const canRun = ready && created && !loadingModel;

  return (
    <div className="section" style={{ minHeight: "100vh" }}>
      <div
        className="container is-fluid"
        style={{ maxWidth: "100%", padding: "0 2rem" }}
      >
        <h1 className="title is-3">Paragon MNIST • Portal</h1>

        {/* Controls */}
        <div className="box">
          <div className="field is-grouped is-grouped-multiline">
            <p className="control">
              <button
                className={`button is-primary ${created ? "is-static" : ""}`}
                onClick={onCreate}
                disabled={disabled || created}
              >
                {created ? "Net Ready" : "Create Net"}
              </button>
            </p>
            <p className="control">
              <button
                className={`button is-link ${loadingModel ? "is-loading" : ""}`}
                onClick={onLoad}
                disabled={!created || loadingModel}
              >
                Load Model
              </button>
            </p>

            <p className="control">
              <span className="select">
                <select
                  value={digit}
                  onChange={(e) => setDigit(Number(e.target.value))}
                >
                  {CLASSES.map((d) => (
                    <option key={d} value={d}>
                      digit {d}
                    </option>
                  ))}
                </select>
              </span>
            </p>

            <p className="control">
              <label className="checkbox">
                <input
                  type="checkbox"
                  className="mr-2"
                  checked={invert}
                  onChange={(e) => setInvert(e.target.checked)}
                />
                Invert input
              </label>
            </p>

            <p className="control">
              <label className="checkbox">
                <input
                  type="checkbox"
                  className="mr-2"
                  checked={clientSoftmax}
                  onChange={(e) => setClientSoftmax(e.target.checked)}
                />
                Client softmax
              </label>
            </p>

            <p className="control">
              <button className="button is-info" onClick={refreshAll}>
                Refresh All
              </button>
            </p>

            <p className="control">
              <button className="button is-light" onClick={copyCSV}>
                Copy CSV
              </button>
            </p>
          </div>

          <p className="help">
            Services refresh independently. The table shows probabilities (10
            d.p.) for the selected digit across all services. Portal uses images
            from <code>{nodeUrl}</code>.
          </p>
        </div>

        {/* Per-service refresh + status */}
        <div className="box">
          <p className="title is-6">Services</p>
          <ul>
            {services.map((s) => (
              <li key={s.label} className="mb-2">
                <code>{s.label}</code> →{" "}
                <code>{s.kind === "http" ? s.url : "browser"}</code>
                <button
                  className={`button is-small is-light ml-2 ${
                    loadingService === s.label ? "is-loading" : ""
                  }`}
                  onClick={() => refreshService(s.label)}
                  disabled={loadingService !== null}
                >
                  Refresh
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Minimal, professional table: service, digit, p0..p9 */}
        <div
          className="table-container has-background-dark"
          style={{
            width: "80vw",
            margin: "0 auto",
            border: "1px solid #333",
            borderRadius: "8px",
            boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
            overflowX: "auto",
          }}
        >
          <table
            className="table is-fullwidth is-hoverable is-narrow"
            style={{
              color: "#e0e0e0",
              backgroundColor: "#1e1e1e",
              minWidth: "900px",
              borderCollapse: "separate",
              borderSpacing: "0",
            }}
          >
            <thead>
              <tr style={{ backgroundColor: "#2a2a2a" }}>
                <th
                  style={{
                    position: "sticky",
                    left: 0,
                    backgroundColor: "#2a2a2a",
                    zIndex: 2,
                    color: "#ffffff",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  service
                </th>
                <th style={{ color: "#ffffff" }}>digit</th>
                {CLASSES.map((k) => (
                  <th
                    key={`p${k}`}
                    style={{
                      textAlign: "right",
                      whiteSpace: "nowrap",
                      color: "#ffffff",
                    }}
                  >
                    p{k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {services.map((s, rowIdx) => {
                const probs = rows[s.label];
                const rowBg = rowIdx % 2 === 0 ? "#1f1f1f" : "#262626"; // subtle zebra striping
                return (
                  <tr key={s.label} style={{ backgroundColor: rowBg }}>
                    <td
                      style={{
                        position: "sticky",
                        left: 0,
                        backgroundColor: rowBg,
                        zIndex: 1,
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        color: "#00d1b2", // Bulma turquoise accent
                      }}
                    >
                      {s.label}
                    </td>
                    <td style={{ color: "#e0e0e0" }}>{digit}</td>
                    {CLASSES.map((k) => (
                      <td
                        key={`${s.label}-p${k}`}
                        style={{
                          fontFamily:
                            "ui-monospace, Menlo, Consolas, monospace",
                          textAlign: "right",
                          whiteSpace: "nowrap",
                          color: "#e0e0e0",
                        }}
                      >
                        {probs && probs.length === 10
                          ? probs[k].toFixed(10)
                          : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer hint */}
        <p className="help mt-3">
          Tip: after restarting any microservice, press its <em>Refresh</em> to
          repopulate that row for the selected digit.
        </p>
      </div>
    </div>
  );
}
