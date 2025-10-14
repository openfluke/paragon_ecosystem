package main

import (
	"embed"
	"log"
	"net/http"
	"os"
	"strings"
)

//go:embed public/*
var staticFS embed.FS

func main() {
	addr := getenv("ADDR", ":8009") // where this Go server listens
	// This serves files from ./public at the root ("/")
	fs := http.FileServer(http.FS(staticFS))
	http.Handle("/", fs)

	log.Printf("Serving vanilla portal UI on http://127.0.0.1%v\n", addr)
	log.Printf("Tip: Your ML service base URL can be set in the page UI (defaults to http://127.0.0.1:8001)")
	log.Fatal(http.ListenAndServe(addr, nil))
}

func getenv(k, def string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return def
}
