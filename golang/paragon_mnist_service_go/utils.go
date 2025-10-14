package main

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	mnistBase   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	trainImgsGZ = "train-images-idx3-ubyte.gz"
	trainLabsGZ = "train-labels-idx1-ubyte.gz"
)

func getEnv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func ensureDir(p string) error {
	return os.MkdirAll(p, 0o755)
}

func fileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, err
}

func downloadFile(url, outPath string) error {
	if ok, _ := fileExists(outPath); ok {
		return nil
	}
	if err := ensureDir(filepath.Dir(outPath)); err != nil {
		return err
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return errors.New(resp.Status)
	}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func unzipGZToFile(gzPath, rawPath string) error {
	if ok, _ := fileExists(rawPath); ok {
		return nil
	}
	in, err := os.Open(gzPath)
	if err != nil {
		return err
	}
	defer in.Close()
	gr, err := gzip.NewReader(in)
	if err != nil {
		return err
	}
	defer gr.Close()
	out, err := os.Create(rawPath)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, gr)
	return err
}

func autopopulateImages() error {
	// if any PNG already exists, skip
	entries, _ := os.ReadDir(imagesDir)
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(stringsLower(e.Name())) == ".png" {
			return nil
		}
	}
	// download + extract MNIST idx files
	mnistDir := "./mnist_idx"
	if err := ensureDir(mnistDir); err != nil {
		return err
	}

	imgGZ := filepath.Join(mnistDir, trainImgsGZ)
	labGZ := filepath.Join(mnistDir, trainLabsGZ)
	if err := downloadFile(mnistBase+trainImgsGZ, imgGZ); err != nil {
		return err
	}
	if err := downloadFile(mnistBase+trainLabsGZ, labGZ); err != nil {
		return err
	}

	imgRaw := filepath.Join(mnistDir, "train-images-idx3-ubyte")
	labRaw := filepath.Join(mnistDir, "train-labels-idx1-ubyte")
	if err := unzipGZToFile(imgGZ, imgRaw); err != nil {
		return err
	}
	if err := unzipGZToFile(labGZ, labRaw); err != nil {
		return err
	}

	images, err := readImagesIDX(imgRaw)
	if err != nil {
		return err
	}
	labels, err := readLabelsIDX(labRaw)
	if err != nil {
		return err
	}

	seen := map[int]bool{}
	for i := 0; i < len(images) && len(seen) < 10; i++ {
		lbl := labels[i]
		if seen[lbl] {
			continue
		}
		if err := writePNG28x28(filepath.Join(imagesDir, strconv.Itoa(lbl)+".png"), images[i]); err != nil {
			return err
		}
		seen[lbl] = true
	}
	return nil
}

func readImagesIDX(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var head [16]byte
	if _, err := io.ReadFull(f, head[:]); err != nil {
		return nil, err
	}
	magic := binary.BigEndian.Uint32(head[0:4])
	if magic != 2051 {
		return nil, errors.New("bad magic for images")
	}
	num := int(binary.BigEndian.Uint32(head[4:8]))
	rows := int(binary.BigEndian.Uint32(head[8:12]))
	cols := int(binary.BigEndian.Uint32(head[12:16]))

	images := make([][][]float64, num)
	buf := make([]byte, rows*cols)
	for i := 0; i < num; i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, err
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			row := make([]float64, cols)
			for c := 0; c < cols; c++ {
				row[c] = float64(buf[r*cols+c]) / 255.0
			}
			img[r] = row
		}
		images[i] = img
	}
	return images, nil
}

func readLabelsIDX(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var head [8]byte
	if _, err := io.ReadFull(f, head[:]); err != nil {
		return nil, err
	}
	magic := binary.BigEndian.Uint32(head[0:4])
	if magic != 2049 {
		return nil, errors.New("bad magic for labels")
	}
	num := int(binary.BigEndian.Uint32(head[4:8]))
	labels := make([]int, num)
	b := make([]byte, 1)
	for i := 0; i < num; i++ {
		if _, err := io.ReadFull(f, b); err != nil {
			return nil, err
		}
		labels[i] = int(b[0])
	}
	return labels, nil
}

func writePNG28x28(outPath string, img [][]float64) error {
	if err := ensureDir(filepath.Dir(outPath)); err != nil {
		return err
	}
	h := len(img)
	w := len(img[0])
	gray := image.NewGray(image.Rect(0, 0, w, h))
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			v := uint8(img[r][c] * 255.0)
			gray.SetGray(c, r, color.Gray{Y: v})
		}
	}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, gray)
}

func loadPNG28x28(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	im, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	if w != 28 || h != 28 {
		// normalize to 28x28 if someone drops a different PNG in
		dst := image.NewGray(image.Rect(0, 0, 28, 28))
		// nearest-neighbor manual scale
		for y := 0; y < 28; y++ {
			for x := 0; x < 28; x++ {
				sx := b.Min.X + x*w/28
				sy := b.Min.Y + y*h/28
				R, G, B, _ := im.At(sx, sy).RGBA()
				Y := (0.2126*float64(R) + 0.7152*float64(G) + 0.0722*float64(B)) / 65535.0
				dst.SetGray(x, y, color.Gray{Y: uint8(Y*255 + 0.5)})
			}
		}
		// convert dst back to [][]float64
		out := make([][]float64, 28)
		for r := 0; r < 28; r++ {
			row := make([]float64, 28)
			for c := 0; c < 28; c++ {
				row[c] = float64(dst.GrayAt(c, r).Y) / 255.0
			}
			out[r] = row
		}
		return out, nil
	}
	// exact 28x28
	out := make([][]float64, 28)
	for r := 0; r < 28; r++ {
		row := make([]float64, 28)
		for c := 0; c < 28; c++ {
			R, G, B, _ := im.At(b.Min.X+c, b.Min.Y+r).RGBA()
			Y := (0.2126*float64(R) + 0.7152*float64(G) + 0.0722*float64(B)) / 65535.0
			row[c] = Y
		}
		out[r] = row
	}
	return out, nil
}

func listImages() ([]string, error) {
	ents, err := os.ReadDir(imagesDir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if filepath.Ext(stringsLower(e.Name())) == ".png" {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

func stringsLower(s string) string { return strings.ToLower(s) }
