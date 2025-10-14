package main

import (
	"errors"
	"math"
	"time"

	"github.com/openfluke/paragon/v3"
)

type ParagonHandle struct {
	nn *paragon.Network[float32]
}

func initializeModels(modelPath string) (*ParagonHandle, *ParagonHandle, bool, error) {
	// Create a minimal model if missing
	if ok, _ := fileExists(modelPath); !ok {
		if err := createDefaultModelJSON(modelPath); err != nil {
			return nil, nil, false, err
		}
	}

	// Load JSON (type-aware), then reconstruct float32 net and copy weights
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(modelPath)
	if err != nil {
		return nil, nil, false, err
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		return nil, nil, false, errors.New("model is not float32")
	}
	shapes, activs, trainable := topologyFrom(tmp)

	// CPU handle
	nnCPU, err := paragon.NewNetwork[float32](shapes, activs, trainable)
	if err != nil {
		return nil, nil, false, err
	}
	state, _ := tmp.MarshalJSONModel()
	if err := nnCPU.UnmarshalJSONModel(state); err != nil {
		return nil, nil, false, err
	}

	// GPU handle (optional)
	nnGPU, err := paragon.NewNetwork[float32](shapes, activs, trainable)
	if err != nil {
		return nil, nil, false, err
	}
	if err := nnGPU.UnmarshalJSONModel(state); err != nil {
		return nil, nil, false, err
	}
	nnGPU.WebGPUNative = true

	gpuOK := true
	start := time.Now()
	if err := nnGPU.InitializeOptimizedGPU(); err != nil {
		// fall back to CPU-only if GPU init fails
		gpuOK = false
		nnGPU.WebGPUNative = false
	} else {
		_ = warmupGPU(nnGPU)
	}
	_ = start

	return &ParagonHandle{nnCPU}, &ParagonHandle{nnGPU}, gpuOK, nil
}

func warmupGPU(nn *paragon.Network[float32]) error {
	// 28x28 zeros just to compile pipeline once
	img := make([][]float64, 28)
	for r := 0; r < 28; r++ {
		row := make([]float64, 28)
		img[r] = row
	}
	nn.Forward(img)
	_ = nn.ExtractOutput()
	return nil
}

func (h *ParagonHandle) Forward(img [][]float64) {
	h.nn.Forward(img)
}
func (h *ParagonHandle) ExtractOutput() []float64 {
	return h.nn.ExtractOutput()
}

func forwardProbs(h *ParagonHandle, img [][]float64) (*ProbResult, error) {
	h.Forward(img)
	logits := h.ExtractOutput()

	// last 10 entries are logits for classes
	if len(logits) < 10 {
		return nil, errors.New("output too small")
	}
	start := len(logits) - 10
	probs := softmax(logits[start:])
	pred := argmax(probs)
	return &ProbResult{Pred: pred, Probs: probs}, nil
}

func softmax(x []float64) []float64 {
	maxv := x[0]
	for _, v := range x[1:] {
		if v > maxv {
			maxv = v
		}
	}
	exp := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		e := math.Exp(v - maxv)
		exp[i] = e
		sum += e
	}
	for i := range exp {
		exp[i] /= sum
	}
	return exp
}

func argmax(v []float64) int {
	best, idx := v[0], 0
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best, idx = v[i], i
		}
	}
	return idx
}

// Best-effort topology extraction; keeps the same layer shapes/activations/trainable
func topologyFrom(tmp *paragon.Network[float32]) ([]struct{ Width, Height int }, []string, []bool) {
	n := len(tmp.Layers)
	shapes := make([]struct{ Width, Height int }, n)
	acts := make([]string, n)
	tr := make([]bool, n)

	for i, L := range tmp.Layers {
		shapes[i] = struct{ Width, Height int }{L.Width, L.Height}
		act := "linear"
		if L.Height > 0 && L.Width > 0 && L.Neurons != nil && len(L.Neurons) > 0 && len(L.Neurons[0]) > 0 && L.Neurons[0][0] != nil {
			act = L.Neurons[0][0].Activation
		}
		acts[i], tr[i] = act, true
	}
	return shapes, acts, tr
}

func createDefaultModelJSON(path string) error {
	// shapes [(28,28), (256,1), (10,1)] with activations ["linear","relu","softmax"]
	shapes := []struct{ Width, Height int }{
		{28, 28}, {256, 1}, {10, 1},
	}
	acts := []string{"linear", "relu", "softmax"}
	train := []bool{true, true, true}

	nn, err := paragon.NewNetwork[float32](shapes, acts, train)
	if err != nil {
		return err
	}
	return nn.SaveJSON(path)
}

func round6(x float64) float64 { return math.Round(x*1e6) / 1e6 }
