package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	defaultTimeout   = 120 * time.Second
	maxErrorBodySize = 4096
)

var defaultBackoff = []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second}

// Client talks to the Ollama HTTP API.
type Client struct {
	baseURL string
	model   string
	http    *http.Client
	backoff []time.Duration
}

// Option customizes a Client.
type Option func(*Client)

// WithBackoff sets the pauses between retries; len(d) is the maximum number of retries.
func WithBackoff(d ...time.Duration) Option {
	return func(c *Client) { c.backoff = d }
}

func New(baseURL, model string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		http:    &http.Client{Timeout: defaultTimeout},
		backoff: defaultBackoff,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed returns one embedding per text, in the same order.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(embedRequest{Model: c.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("encode embed request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		embeddings, retryable, err := c.embedOnce(ctx, body, len(texts))
		if err == nil {
			return embeddings, nil
		}
		if !retryable || attempt >= len(c.backoff) {
			return nil, err
		}

		timer := time.NewTimer(c.backoff[attempt])
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, fmt.Errorf("wait before retry (last error: %v): %w", err, ctx.Err())
		case <-timer.C:
		}
	}
}

// embedOnce performs a single request. The bool reports whether a failure is worth retrying.
func (c *Client) embedOnce(ctx context.Context, body []byte, want int) ([][]float32, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		// A cancelled or expired ctx will not get better on retry.
		retryable := ctx.Err() == nil
		return nil, retryable, fmt.Errorf("ollama embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode >= 500, statusError(resp)
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, false, fmt.Errorf("decode embed response: %w", err)
	}
	if len(out.Embeddings) != want {
		return nil, false, fmt.Errorf("ollama returned %d embeddings for %d texts", len(out.Embeddings), want)
	}
	return out.Embeddings, false, nil
}

func statusError(resp *http.Response) error {
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
	if err != nil {
		return fmt.Errorf("ollama: status %d (read error body: %w)", resp.StatusCode, err)
	}

	msg := strings.TrimSpace(string(raw))
	var parsed struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(raw, &parsed) == nil && parsed.Error != "" {
		msg = parsed.Error
	}
	if msg == "" {
		return errors.New("ollama: status " + resp.Status)
	}
	return fmt.Errorf("ollama: status %d: %s", resp.StatusCode, msg)
}
