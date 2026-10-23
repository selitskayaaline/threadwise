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
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 0, Channels: []string{"c"}}); err == nil {
		t.Fatal("expected error for k=0")
	}
}

func TestSearchEmbedError(t *testing.T) {
	boom := errors.New("boom")
	v := NewVector(nil, fakeEmbedder{err: boom})
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 1, Channels: []string{"c"}}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapping %v", err, boom)
	}
}

func TestSearchWrongDimension(t *testing.T) {
	v := NewVector(nil, fakeEmbedder{vec: make([]float32, 3)})
	if _, err := v.Search(context.Background(), Query{Text: "q", K: 1, Channels: []string{"c"}}); err == nil {
		t.Fatal("expected error for wrong dimension")
	}
}

// With no channels both retrievers return nothing before touching the
// database (nil pool) or the embedder (which would fail).
func TestSearchWithoutChannelsReturnsNothing(t *testing.T) {
	retrievers := map[string]Retriever{
		"vector": NewVector(nil, fakeEmbedder{err: errors.New("embedder must not be called")}),
		"fts":    NewFTS(nil),
	}
	for name, r := range retrievers {
		for _, channels := range [][]string{nil, {}} {
			res, err := r.Search(context.Background(), Query{Text: "q", K: 5, Channels: channels})
			if err != nil {
				t.Errorf("%s channels=%#v: unexpected error: %v", name, channels, err)
			}
			if len(res) != 0 {
				t.Errorf("%s channels=%#v: got %d results, want none", name, channels, len(res))
			}
		}
	}
}
