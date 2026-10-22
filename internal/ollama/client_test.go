package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const shortPause = time.Millisecond

func newTestClient(url string) *Client {
	return New(url, "bge-m3", WithBackoff(shortPause, shortPause, shortPause))
}

func writeEmbeddings(w http.ResponseWriter, vecs [][]float32) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": vecs})
}

func TestEmbedSuccess(t *testing.T) {
	want := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/embed" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		var req embedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Model != "bge-m3" || !reflect.DeepEqual(req.Input, []string{"a", "b"}) {
			t.Errorf("unexpected request body %+v", req)
		}
		writeEmbeddings(w, want)
	}))
	defer srv.Close()

	got, err := newTestClient(srv.URL).Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestEmbedRetriesOn5xx(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, `{"error":"boom"}`, http.StatusInternalServerError)
			return
		}
		writeEmbeddings(w, [][]float32{{1}})
	}))
	defer srv.Close()

	got, err := newTestClient(srv.URL).Embed(context.Background(), []string{"a"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("got %d vectors, want 1", len(got))
	}
	if n := calls.Load(); n != 2 {
		t.Errorf("got %d requests, want 2", n)
	}
}

func TestEmbedNoRetryOn4xx(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer srv.Close()

	_, err := newTestClient(srv.URL).Embed(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "400") || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("error %q lacks status or Ollama message", err)
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("got %d requests, want 1", n)
	}
}

func TestEmbedCountMismatch(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		writeEmbeddings(w, [][]float32{{1}})
	}))
	defer srv.Close()

	_, err := newTestClient(srv.URL).Embed(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected error")
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("got %d requests, want 1", n)
	}
}

func TestEmbedRetriesExhausted(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	_, err := newTestClient(srv.URL).Embed(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
	if n := calls.Load(); n != 4 {
		t.Errorf("got %d requests, want 4 (1 + 3 retries)", n)
	}
}

func TestEmbedContextCancelledDuringPause(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "down", http.StatusInternalServerError)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	c := New(srv.URL, "bge-m3", WithBackoff(time.Hour))
	start := time.Now()
	_, err := c.Embed(ctx, []string{"a"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want context.DeadlineExceeded", err)
	}
	if time.Since(start) > 5*time.Second {
		t.Error("Embed did not return promptly after ctx expired")
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("got %d requests, want 1", n)
	}
}

func TestChatSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/chat" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Model != "gemma3:4b" || req.Stream || req.Options.Temperature != 0.1 {
			t.Errorf("unexpected request body %+v", req)
		}
		wantMsgs := []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "u"}}
		if !reflect.DeepEqual(req.Messages, wantMsgs) {
			t.Errorf("messages = %+v, want %+v", req.Messages, wantMsgs)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"message": Message{Role: "assistant", Content: "ответ [1]"}})
	}))
	defer srv.Close()

	c := New(srv.URL, "gemma3:4b", WithBackoff(shortPause))
	got, err := c.Chat(context.Background(), []Message{{Role: "system", Content: "s"}, {Role: "user", Content: "u"}}, 0.1)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "ответ [1]" {
		t.Errorf("got %q", got)
	}
}

func TestChatRetriesOn5xx(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, `{"error":"boom"}`, http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"message": Message{Role: "assistant", Content: "ok"}})
	}))
	defer srv.Close()

	got, err := newTestClient(srv.URL).Chat(context.Background(), []Message{{Role: "user", Content: "u"}}, 0)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "ok" || calls.Load() != 2 {
		t.Errorf("got %q after %d requests, want ok after 2", got, calls.Load())
	}
}

func TestChatNoRetryOn4xx(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer srv.Close()

	_, err := newTestClient(srv.URL).Chat(context.Background(), []Message{{Role: "user", Content: "u"}}, 0)
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("got %v, want error with Ollama message", err)
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("got %d requests, want 1", n)
	}
}

func TestChatEmptyAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"  "}}`))
	}))
	defer srv.Close()

	if _, err := newTestClient(srv.URL).Chat(context.Background(), []Message{{Role: "user", Content: "u"}}, 0); err == nil {
		t.Fatal("expected error for empty answer")
	}
}

func TestChatTimeoutLongerThanEmbed(t *testing.T) {
	c := New("http://x", "m")
	if c.chat.Timeout <= c.http.Timeout {
		t.Errorf("chat timeout %v must exceed embed timeout %v", c.chat.Timeout, c.http.Timeout)
	}
}
