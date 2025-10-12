// bench_paragon.go — Simple Paragon CPU vs GPU Benchmark (Go)
//
// Usage:
//   go mod tidy
//   go run ./bench_paragon.go               # verbose (prints outputs & per-index diffs)
//   go run ./bench_paragon.go --quiet       # quiet summary only
//   go run ./bench_paragon.go --csv out.csv # write CSV rows (append) in quiet or verbose
//
// Backend hint (optional):
//   WGPU_BACKEND=vulkan go run ./bench_paragon.go --quiet
//
// Env var to point to headless displays if needed (Linux):
//   DISPLAY=:0 WGPU_BACKEND=gl go run ./bench_paragon.go --quiet
//
// Notes:
// - Input is a deterministic 1×784 row (like the C#/Python versions).
// - Shapes: [(784,1) -> ... -> (10,1)] with linear / relu / softmax activations.

package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/openfluke/paragon/v3"
)

type caseShape struct {
	ID     string
	Layers []int
}

var mnistZoo = []caseShape{
	{"S1", []int{784, 64, 10}},
	{"S2", []int{784, 128, 10}},
	{"S3", []int{784, 256, 10}},
	{"M1", []int{784, 256, 256, 10}},
	{"M2", []int{784, 384, 384, 10}},
	{"M3", []int{784, 512, 512, 10}},
	{"L1", []int{784, 768, 768, 768, 10}},
	{"L2", []int{784, 1024, 1024, 1024, 10}},
	{"XL1", []int{784, 1536, 1536, 1536, 1536, 10}},
	{"XL2", []int{784, 2048, 2048, 2048, 2048, 10}},
}

func shapeStr(s caseShape) string {
	parts := make([]string, len(s.Layers))
	for i, n := range s.Layers {
		parts[i] = fmt.Sprintf("%d", n)
	}
	return strings.Join(parts, " → ")
}

func estimateVramMB(s caseShape) float64 {
	L := s.Layers
	var params int64
	for i := 0; i < len(L)-1; i++ {
		params += int64(L[i]) * int64(L[i+1]) // weights
	}
	for i := 1; i < len(L); i++ {
		params += int64(L[i]) // biases
	}
	return float64(params) * 4.0 / (1024 * 1024) // float32
}

func buildParagonShapes(s caseShape) []struct{ Width, Height int } {
	ps := make([]struct{ Width, Height int }, 0, len(s.Layers))
	for i := range s.Layers {
		if i == 0 {
			ps = append(ps, struct{ Width, Height int }{784, 1})
		} else if i == len(s.Layers)-1 {
			ps = append(ps, struct{ Width, Height int }{10, 1})
		} else {
			ps = append(ps, struct{ Width, Height int }{s.Layers[i], 1})
		}
	}
	return ps
}

func buildActivations(s caseShape) []string {
	acts := make([]string, 0, len(s.Layers))
	for i := range s.Layers {
		switch {
		case i == 0:
			acts = append(acts, "linear")
		case i == len(s.Layers)-1:
			acts = append(acts, "softmax")
		default:
			acts = append(acts, "relu")
		}
	}
	return acts
}

func buildTrainable(n int) []bool {
	tb := make([]bool, n)
	for i := range tb {
		tb[i] = true
	}
	return tb
}

// deterministic 1×784 row ( [][]float64 with 1 row )
func fixedRow784(seed uint32) [][]float64 {
	next := func(s *uint32) float64 {
		*s = *s*1664525 + 1013904223
		return float64(*s) / float64(^uint32(0))
	}
	row := make([]float64, 784)
	for i := 0; i < 784; i++ {
		row[i] = math.Round(next(&seed)*1e6) / 1e6
	}
	return [][]float64{row}
}

type forwardOut struct {
	ms   float64
	raw  []float64
	flat []float64
}

// time a single Forward+ExtractOutput
func forwardTimed(nn *paragon.Network[float32], input [][]float64) forwardOut {
	start := time.Now()
	nn.Forward(input)
	out := nn.ExtractOutput() // []float64
	elapsed := time.Since(start).Seconds() * 1000.0
	return forwardOut{ms: elapsed, raw: out, flat: out}
}

func diffStats(a, b []float64) (mae, maxd float64, n int) {
	n = min(len(a), len(b))
	if n == 0 {
		return 0, 0, 0
	}
	var sum, maxAbs float64
	for i := 0; i < n; i++ {
		d := math.Abs(a[i] - b[i])
		sum += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	return sum / float64(n), maxAbs, n
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func printVector(label string, v []float64) {
	fmt.Printf("%s: [", label)
	for i, x := range v {
		if i > 0 {
			fmt.Print(", ")
		}
		// compact-ish scientific for tiny/huge values
		if (math.Abs(x) > 1e4) || (math.Abs(x) > 0 && math.Abs(x) < 1e-3) {
			fmt.Printf("%.6g", x)
		} else {
			fmt.Printf("%.9g", x)
		}
	}
	fmt.Println("]")
}

type benchRow struct {
	ID       string
	Shape    string
	EstMB    float64
	CPUms    float64
	GPUms    float64
	Speedup  float64
	MAE      float64
	Max      float64
	InitMS   float64
	Adapter  string
	Enabled  bool
	OutCPU   []float64
	OutGPU   []float64
	InputHex string // optional placeholder if you ever serialize inputs
}

func runCase(spec caseShape, quiet bool) benchRow {
	fmt.Printf("\n=== %s (%s) ===\n", spec.ID, shapeStr(spec))
	seed := uint32(123)
	x := fixedRow784(seed)

	// Build fresh network
	nn, err := paragon.NewNetwork[float32](buildParagonShapes(spec), buildActivations(spec), buildTrainable(len(spec.Layers)))
	if err != nil {
		fmt.Println("NewNetwork failed:", err)
		return benchRow{ID: spec.ID, Shape: shapeStr(spec)}
	}
	nn.Debug = false

	// CPU warmup
	nn.WebGPUNative = false
	nn.Forward(x)
	_ = nn.ExtractOutput()
	cpu := forwardTimed(nn, x)

	// GPU init
	nn.WebGPUNative = true
	startInit := time.Now()
	err = nn.InitializeOptimizedGPU()
	initMS := time.Since(startInit).Seconds() * 1000.0
	enabled := true
	adapter := "[ok]"
	if err != nil {
		adapter = "error:" + err.Error()
		enabled = false
		nn.WebGPUNative = false
	}
	fmt.Printf("GPU init: %s  in %.2f ms  enabled=%s\n", adapter, initMS, map[bool]string{true: "yes", false: "no"}[enabled])

	// Warmup on GPU (or CPU fallback)
	nn.Forward(x)
	_ = nn.ExtractOutput()
	gpu := forwardTimed(nn, x)

	mae, maxd, n := diffStats(cpu.flat, gpu.flat)

	// logs
	fmt.Printf("CPU  ⏱ %.3f ms\n", cpu.ms)
	fmt.Printf("GPU  ⏱ %.3f ms\n", gpu.ms)
	speed := math.Inf(1)
	if gpu.ms > 0 {
		speed = cpu.ms / gpu.ms
	}
	fmt.Printf("Speedup: %.2fx\n", speed)
	fmt.Printf("Δ(CPU vs GPU)  mae=%.2E  max=%.2E  (n=%d)\n", mae, maxd, n)

	if !quiet {
		printVector("CPU ExtractOutput (raw)", cpu.raw)
		printVector("GPU ExtractOutput (raw)", gpu.raw)

		// quick softmax view when the head is 10-wide
		if len(cpu.raw) == 10 {
			fmt.Printf("%-4s| %-22s | %-22s | %-s\n", "Idx", "CPU", "GPU", "Δ")
			fmt.Println("----+------------------------+------------------------+------------------")
			for i := 0; i < 10; i++ {
				fmt.Printf("%3d | %22.16g | %22.16g | %16.9e\n", i, cpu.raw[i], gpu.raw[i], math.Abs(cpu.raw[i]-gpu.raw[i]))
			}
		}
	}

	// cleanup
	if enabled {
		nn.CleanupOptimizedGPU()
	}

	return benchRow{
		ID:      spec.ID,
		Shape:   shapeStr(spec),
		EstMB:   estimateVramMB(spec),
		CPUms:   cpu.ms,
		GPUms:   gpu.ms,
		Speedup: speed,
		MAE:     mae,
		Max:     maxd,
		InitMS:  initMS,
		Adapter: adapter,
		Enabled: enabled,
		OutCPU:  cpu.raw,
		OutGPU:  gpu.raw,
	}
}

func appendCSV(path string, rows []benchRow) error {
	newFile := false
	if _, err := os.Stat(path); os.IsNotExist(err) {
		newFile = true
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if newFile {
		_ = w.Write([]string{"id", "shape", "estMB", "cpu_ms", "gpu_ms", "speedup", "mae", "max", "gpu_init_ms", "adapter"})
	}
	for _, r := range rows {
		rec := []string{
			r.ID,
			r.Shape,
			fmt.Sprintf("%.2f", r.EstMB),
			fmt.Sprintf("%.3f", r.CPUms),
			fmt.Sprintf("%.3f", r.GPUms),
			fmt.Sprintf("%.2f", r.Speedup),
			fmt.Sprintf("%.2E", r.MAE),
			fmt.Sprintf("%.2E", r.Max),
			fmt.Sprintf("%.2f", r.InitMS),
			r.Adapter,
		}
		_ = w.Write(rec)
	}
	w.Flush()
	return w.Error()
}

func main() {
	quiet := flag.Bool("quiet", false, "suppress per-index vectors")
	csvPath := flag.String("csv", "", "append results to CSV file")
	flag.Parse()

	fmt.Println("Simple Paragon CPU vs GPU Benchmark (Go)")
	fmt.Println("========================================")

	results := make([]benchRow, 0, len(mnistZoo))
	for _, spec := range mnistZoo {
		r := runCase(spec, *quiet)
		results = append(results, r)
	}

	if *csvPath != "" {
		if err := appendCSV(*csvPath, results); err != nil {
			fmt.Println("CSV write error:", err)
		} else {
			fmt.Println("💾 CSV appended →", *csvPath)
		}
	}
}
