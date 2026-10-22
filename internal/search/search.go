package search

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/store"
)

// Result is a message found by a Retriever. Score depends on the Retriever:
// for Vector it is the cosine distance (the smaller, the closer), for FTS it
// is ts_rank (the larger, the better). Results come best first.
type Result struct {
	ID      int
	Channel string
	Author  string
	TS      time.Time
	Text    string
	Score   float64
}

// Retriever finds the k messages most relevant to query.
type Retriever interface {
	Search(ctx context.Context, query string, k int) ([]Result, error)
}

// Embedder turns texts into vectors; *ollama.Client satisfies it.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Vector is a Retriever backed by pgvector cosine distance.
type Vector struct {
	pool *pgxpool.Pool
	emb  Embedder
}

func NewVector(pool *pgxpool.Pool, emb Embedder) *Vector {
	return &Vector{pool: pool, emb: emb}
}

const searchSQL = `
SELECT id, channel, author, ts, text, embedding <=> $1::vector
FROM messages
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2`

func (v *Vector) Search(ctx context.Context, query string, k int) ([]Result, error) {
	if k < 1 {
		return nil, fmt.Errorf("search: k must be at least 1, got %d", k)
	}

	vecs, err := v.emb.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embed query: got %d vectors, want 1", len(vecs))
	}
	if len(vecs[0]) != store.EmbeddingDim {
		return nil, fmt.Errorf("embed query: got dimension %d, want %d", len(vecs[0]), store.EmbeddingDim)
	}

	rows, err := v.pool.Query(ctx, searchSQL, store.FormatVector(vecs[0]), k)
	if err != nil {
		return nil, fmt.Errorf("query nearest messages: %w", err)
	}
	defer rows.Close()

	var res []Result
	for rows.Next() {
		var r Result
		if err := rows.Scan(&r.ID, &r.Channel, &r.Author, &r.TS, &r.Text, &r.Score); err != nil {
			return nil, fmt.Errorf("scan result: %w", err)
		}
		res = append(res, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read results: %w", err)
	}
	return res, nil
}
