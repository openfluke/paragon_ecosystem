import React, { useEffect, useState } from "react";
import { initPortal } from "@openfluke/portal";

export default function PortalDemo() {
  const [portal, setPortal] = useState<any>(null);
  const [output, setOutput] = useState<string>("Loading…");

  useEffect(() => {
    (async () => {
      const api = await initPortal();
      setPortal(api);

      const nn = api.NewNetworkFloat32(
        JSON.stringify([
          { Width: 1, Height: 1 },
          { Width: 2, Height: 1 },
          { Width: 3, Height: 1 },
        ]),
        JSON.stringify(["linear", "relu", "softmax"]),
        JSON.stringify([true, true, true])
      );

      nn.PerturbWeights(JSON.stringify([0.1, Date.now() % 1000]));
      nn.Forward(JSON.stringify([[[Math.random()]]]));
      setOutput(nn.ExtractOutput());
    })();
  }, []);

  return (
    <div>
      <h1>Portal Demo</h1>
      <pre>{output}</pre>
    </div>
  );
}
