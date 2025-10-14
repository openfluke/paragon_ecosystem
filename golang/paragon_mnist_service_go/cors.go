package main

import (
	"encoding/json"
	"net/http"
)

// permissive CORS like your FastAPI setup; tighten in prod
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

type httpError struct {
	code int
	msg  string
}

func newHTTPError(code int, msg string) *httpError { return &httpError{code, msg} }
func (e *httpError) Error() string                 { return e.msg }
func httpStatus(err error) int {
	if he, ok := err.(*httpError); ok {
		return he.code
	}
	return http.StatusInternalServerError
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
