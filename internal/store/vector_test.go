package store

import (
	"strings"
	"testing"
)

func TestFormatVector(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want string
	}{
		{"empty", []float32{}, "[]"},
		{"single", []float32{1}, "[1]"},
		{"mixed", []float32{0.1, -0.25, 0, 3}, "[0.1,-0.25,0,3]"},
		{"small", []float32{0.00001}, "[1e-05]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FormatVector(tt.in); got != tt.want {
				t.Errorf("FormatVector(%v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestCheckDim(t *testing.T) {
	if err := checkDim(1, make([]float32, EmbeddingDim)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err := checkDim(7, make([]float32, 768))
	if err == nil {
		t.Fatal("expected error")
	}
	for _, want := range []string{"модель вернула 768", "колонка ждёт 1024", "id=7"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
}
