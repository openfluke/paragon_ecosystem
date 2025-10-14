package main

import (
	"encoding/json"
	"log"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type PredictRequest struct {
	Image   string `json:"image"`
	Backend string `json:"backend"` // "gpu" | "cpu"
}

type ProbResult struct {
	Pred       int       `json:"pred"`
	Probs      []float64 `json:"probs"`
	LatencySec float64   `json:"latency_sec"`
}

type ParityRow struct {
	Image string      `json:"image"`
	CPU   *ProbResult `json:"cpu,omitempty"`
	GPU   *ProbResult `json:"gpu,omitempty"`
	Match *bool       `json:"match,omitempty"`
	Error string      `json:"error,omitempty"`
}

type ParityReport struct {
	GPUAvailable bool        `json:"gpu_available"`
	Mismatches   int         `json:"mismatches"`
	Total        int         `json:"total"`
	Results      []ParityRow `json:"results"`
}

// globals
var (
	imagesDir = getEnv("IMAGES_DIR", "./images")
	modelJSON = getEnv("MODEL_JSON", "./mnist_paragon_model.json")
	hCPU      *ParagonHandle
	hGPU      *ParagonHandle
	gpuOK     bool
)

func main() {
	// Ensure folders + images
	if err := ensureDir(imagesDir); err != nil {
		log.Fatalf("make images dir: %v", err)
	}
	if err := autopopulateImages(); err != nil {
		log.Printf("⚠️  autopopulate images failed (continuing): %v", err)
	}

	// Init models (CPU + optional GPU)
	var err error
	hCPU, hGPU, gpuOK, err = initializeModels(modelJSON)
	if err != nil {
		log.Fatalf("initialize models: %v", err)
	}

	// Static files for images
	fs := http.FileServer(http.Dir(imagesDir))
	http.Handle("/static/images/", http.StripPrefix("/static/images/", fs))

	// Routes
	http.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"message":       "MNIST service ready (Go)",
			"gpu_available": gpuOK,
		})
	})
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "gpu_available": gpuOK})
	})
	http.HandleFunc("/images/list", func(w http.ResponseWriter, _ *http.Request) {
		imgs, _ := listImages()
		writeJSON(w, http.StatusOK, map[string]any{"images": imgs})
	})

	http.HandleFunc("/predict", handlePredict)        // GET & POST
	http.HandleFunc("/predict-raw", handlePredictRaw) // raw logits endpoint
	http.HandleFunc("/parity", handleParity)

	addr := getEnv("ADDR", "0.0.0.0:8000")
	log.Printf("🚀 Listening on http://%s", addr)
	log.Fatal(http.ListenAndServe(addr, withCORS(http.DefaultServeMux)))
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		image := strings.TrimSpace(r.URL.Query().Get("image"))
		backend := strings.TrimSpace(r.URL.Query().Get("backend"))
		if backend == "" {
			backend = "gpu"
		}
		if image == "" {
			http.Error(w, "missing ?image=", http.StatusBadRequest)
			return
		}
		res, err := predictCore(image, backend)
		if err != nil {
			http.Error(w, err.Error(), httpStatus(err))
			return
		}
		writeJSON(w, http.StatusOK, res)

	case http.MethodPost:
		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if req.Backend == "" {
			req.Backend = "gpu"
		}
		if strings.TrimSpace(req.Image) == "" {
			http.Error(w, "missing image", http.StatusBadRequest)
			return
		}
		res, err := predictCore(req.Image, req.Backend)
		if err != nil {
			http.Error(w, err.Error(), httpStatus(err))
			return
		}
		writeJSON(w, http.StatusOK, res)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func handlePredictRaw(w http.ResponseWriter, r *http.Request) {
	image := strings.TrimSpace(r.URL.Query().Get("image"))
	backend := strings.TrimSpace(r.URL.Query().Get("backend"))
	if backend == "" {
		backend = "gpu"
	}
	if image == "" {
		http.Error(w, "missing ?image=", http.StatusBadRequest)
		return
	}
	path := filepath.Join(imagesDir, image)
	exists, _ := fileExists(path)
	if !exists {
		http.Error(w, "image not found: "+image, http.StatusNotFound)
		return
	}
	img, err := loadPNG28x28(path)
	if err != nil {
		http.Error(w, "bad image: "+err.Error(), http.StatusBadRequest)
		return
	}

	var h *ParagonHandle
	if strings.ToLower(backend) == "gpu" {
		if !gpuOK || hGPU == nil {
			http.Error(w, "GPU backend not available", http.StatusServiceUnavailable)
			return
		}
		h = hGPU
	} else {
		h = hCPU
	}

	// ✅ Forward has no return; ExtractOutput returns only []float64
	h.Forward(img)
	logits := h.ExtractOutput()

	n := len(logits)
	start := 0
	if n >= 10 {
		start = n - 10
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"backend": backend,
		"image":   image,
		"logits":  logits[start:],
	})
}

func handleParity(w http.ResponseWriter, r *http.Request) {
	imgs, _ := listImages()
	if len(imgs) == 0 {
		imgs = []string{"0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"}
	}
	// allow override: /parity?images=0.png&images=1.png
	if qs := r.URL.Query()["images"]; len(qs) > 0 {
		imgs = qs
	}
	sort.Strings(imgs)

	var rows []ParityRow
	mismatches := 0

	for _, name := range imgs {
		path := filepath.Join(imagesDir, name)
		exists, _ := fileExists(path)
		if !exists {
			rows = append(rows, ParityRow{Image: name, Error: "not found"})
			continue
		}
		img, err := loadPNG28x28(path)
		if err != nil {
			rows = append(rows, ParityRow{Image: name, Error: "bad png: " + err.Error()})
			continue
		}

		// CPU
		cpuStart := time.Now()
		cpuOut, err := forwardProbs(hCPU, img)
		if err != nil {
			rows = append(rows, ParityRow{Image: name, Error: "cpu forward: " + err.Error()})
			continue
		}
		cpuOut.LatencySec = round6(time.Since(cpuStart).Seconds())

		// GPU (optional)
		if !gpuOK || hGPU == nil {
			rows = append(rows, ParityRow{Image: name, CPU: cpuOut, GPU: nil, Match: nil})
			continue
		}
		gpuStart := time.Now()
		gpuOut, err := forwardProbs(hGPU, img)
		if err != nil {
			rows = append(rows, ParityRow{Image: name, CPU: cpuOut, Error: "gpu forward: " + err.Error()})
			continue
		}
		gpuOut.LatencySec = round6(time.Since(gpuStart).Seconds())

		m := cpuOut.Pred == gpuOut.Pred
		if !m {
			mismatches++
		}
		rows = append(rows, ParityRow{Image: name, CPU: cpuOut, GPU: gpuOut, Match: &m})
	}

	writeJSON(w, http.StatusOK, ParityReport{
		GPUAvailable: gpuOK,
		Mismatches:   mismatches,
		Total:        len(rows),
		Results:      rows,
	})
}

func predictCore(imageName, backend string) (map[string]any, error) {
	path := filepath.Join(imagesDir, imageName)
	exists, _ := fileExists(path)
	if !exists {
		return nil, newHTTPError(http.StatusNotFound, "image not found: "+imageName)
	}
	img, err := loadPNG28x28(path)
	if err != nil {
		return nil, newHTTPError(http.StatusBadRequest, "bad image: "+err.Error())
	}

	backend = strings.ToLower(strings.TrimSpace(backend))
	target := hCPU
	if backend == "gpu" {
		if !gpuOK || hGPU == nil {
			return nil, newHTTPError(http.StatusServiceUnavailable, "GPU backend not available")
		}
		target = hGPU
	}

	start := time.Now()
	out, err := forwardProbs(target, img)
	if err != nil {
		return nil, newHTTPError(http.StatusInternalServerError, "forward failed: "+err.Error())
	}
	out.LatencySec = round6(time.Since(start).Seconds())

	return map[string]any{
		"backend":          backend,
		"image":            imageName,
		"prediction":       out.Pred,
		"probabilities":    out.Probs,
		"latency_sec":      out.LatencySec,
		"source_image_url": "/static/images/" + imageName,
	}, nil
}
