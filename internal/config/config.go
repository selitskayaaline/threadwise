package config

import (
	"fmt"
	"os"
	"strings"
)

type Config struct {
	DatabaseURL string
	OllamaURL   string
	EmbedModel  string
}

// Load reads the configuration from the environment. Only DATABASE_URL is
// required here; commands that talk to Ollama must also call RequireOllama.
func Load() (Config, error) {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		return Config{}, fmt.Errorf("missing required environment variable DATABASE_URL")
	}
	return Config{
		DatabaseURL: databaseURL,
		OllamaURL:   os.Getenv("OLLAMA_URL"),
		EmbedModel:  os.Getenv("EMBED_MODEL"),
	}, nil
}

// RequireOllama checks the settings needed by commands that call Ollama.
func (c Config) RequireOllama() error {
	var missing []string
	if c.OllamaURL == "" {
		missing = append(missing, "OLLAMA_URL")
	}
	if c.EmbedModel == "" {
		missing = append(missing, "EMBED_MODEL")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required environment variable(s) %s", strings.Join(missing, ", "))
	}
	return nil
}
