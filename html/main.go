package main

import (
	"embed"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

//go:embed public/*
var staticFS embed.FS

func main() {
	addr := getenv("ADDR", ":8009")

	publicDir := "./public"
	useLive := dirExists(publicDir)

	if useLive {
		fmt.Printf("📂 Serving files directly from %s (live reload enabled)\n", publicDir)
		fs := http.FileServer(http.Dir(publicDir))
		http.Handle("/", fs)
	} else {
		fmt.Println("📦 Serving embedded files (no live reload)")
		fs := http.FileServer(http.FS(staticFS))
		http.Handle("/", fs)
	}

	log.Printf("🚀 Vanilla Portal UI on http://127.0.0.1%v\n", addr)
	log.Printf("💡 Tip: ML service base URL can be set in the page UI (defaults to http://127.0.0.1:8001)")
	log.Fatal(http.ListenAndServe(addr, nil))
}

func getenv(k, def string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return def
}

func dirExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}
