package search

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/store"
)

const baseID = 91001

// TestVectorSearch runs against the database from TEST_DATABASE_URL (never
// the dev database). It inserts three messages with ids from 91001 and, on
// cleanup, deletes only those ids.
func TestVectorSearch(t *testing.T) {
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("connect to test database: %v", err)
	}
	t.Cleanup(pool.Close)

	ids := []int{baseID, baseID + 1, baseID + 2}
	cleanup := func() {
		if _, err := pool.Exec(context.Background(), "DELETE FROM messages WHERE id = ANY($1)", ids); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	}
	cleanup()
	t.Cleanup(cleanup)

	ts := time.Date(2026, 10, 5, 9, 0, 0, 0, time.UTC)
	msgs := make([]store.Message, len(ids))
	for i, id := range ids {
		msgs[i] = store.Message{ID: id, Channel: "search-test", Author: "test", Ts: ts, Text: "search test message"}
	}
	if err := store.UpsertMessages(ctx, pool, msgs); err != nil {
		t.Fatalf("UpsertMessages: %v", err)
	}
	for i, id := range ids {
		vec := make([]float32, store.EmbeddingDim)
		vec[i] = 1
		if _, err := pool.Exec(ctx, "UPDATE messages SET embedding = $1::vector WHERE id = $2", store.FormatVector(vec), id); err != nil {
			t.Fatalf("set embedding for id=%d: %v", id, err)
		}
	}

	// Query close to the second message: one on position 1, a little on 0.
	q := make([]float32, store.EmbeddingDim)
	q[1] = 1
	q[0] = 0.1
	v := NewVector(pool, fakeEmbedder{vec: q})

	for _, k := range []int{1, 2, 3} {
		res, err := v.Search(ctx, Query{Text: "question", K: k})
		if err != nil {
			t.Fatalf("Search k=%d: %v", k, err)
		}
		if len(res) == 0 || len(res) > k {
			t.Fatalf("k=%d: got %d results", k, len(res))
		}
		if res[0].ID != baseID+1 {
			t.Errorf("k=%d: first result id = %d, want %d", k, res[0].ID, baseID+1)
		}
		for i := 1; i < len(res); i++ {
			if res[i].Score < res[i-1].Score {
				t.Errorf("k=%d: results not ordered by distance", k)
			}
		}
	}
}
