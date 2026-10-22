package config

import (
	"strings"
	"testing"
)

func TestLoadOnlyDatabaseURL(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgres://x")
	t.Setenv("OLLAMA_URL", "")
	t.Setenv("EMBED_MODEL", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.DatabaseURL != "postgres://x" {
		t.Errorf("DatabaseURL = %q", cfg.DatabaseURL)
	}
}

func TestLoadMissingDatabaseURL(t *testing.T) {
	t.Setenv("DATABASE_URL", "")
	if _, err := Load(); err == nil {
		t.Fatal("expected error")
	}
}

func TestRequireOllama(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr []string
	}{
		{"both set", Config{OllamaURL: "http://o", EmbedModel: "m"}, nil},
		{"url missing", Config{EmbedModel: "m"}, []string{"OLLAMA_URL"}},
		{"model missing", Config{OllamaURL: "http://o"}, []string{"EMBED_MODEL"}},
		{"both missing", Config{}, []string{"OLLAMA_URL", "EMBED_MODEL"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.RequireOllama()
			if len(tt.wantErr) == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
			for _, name := range tt.wantErr {
				if !strings.Contains(err.Error(), name) {
					t.Errorf("error %q does not mention %s", err, name)
				}
			}
		})
	}
}

func TestRequireChat(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr []string
	}{
		{"both set", Config{OllamaURL: "http://o", ChatModel: "m"}, nil},
		{"embed model not needed", Config{OllamaURL: "http://o", ChatModel: "m", EmbedModel: ""}, nil},
		{"url missing", Config{ChatModel: "m"}, []string{"OLLAMA_URL"}},
		{"model missing", Config{OllamaURL: "http://o"}, []string{"CHAT_MODEL"}},
		{"both missing", Config{}, []string{"OLLAMA_URL", "CHAT_MODEL"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.RequireChat()
			if len(tt.wantErr) == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
			for _, name := range tt.wantErr {
				if !strings.Contains(err.Error(), name) {
					t.Errorf("error %q does not mention %s", err, name)
				}
			}
		})
	}
}

func TestLoadReadsChatModel(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgres://x")
	t.Setenv("CHAT_MODEL", "gemma3:4b")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.ChatModel != "gemma3:4b" {
		t.Errorf("ChatModel = %q", cfg.ChatModel)
	}
}
