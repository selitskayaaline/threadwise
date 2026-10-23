package search

import (
	"context"
	"errors"
	"testing"
)

type fakeEmbedder struct {
	vec []float32
	err error
}

func (f fakeEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	return [][]float32{f.vec}, nil
}

func TestSearchRejectsBadK(t *testing.T) {
	v := NewVector(nil, fakeEmbedder{})
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 0}); err == nil {
		t.Fatal("expected error for k=0")
	}
}

func TestSearchEmbedError(t *testing.T) {
	boom := errors.New("boom")
	v := NewVector(nil, fakeEmbedder{err: boom})
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 1}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapping %v", err, boom)
	}
}

func TestSearchWrongDimension(t *testing.T) {
	v := NewVector(nil, fakeEmbedder{vec: make([]float32, 3)})
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 1}); err == nil {
		t.Fatal("expected error for wrong dimension")
	}
}
