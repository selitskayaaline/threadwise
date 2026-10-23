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

const accessBaseID = 93001

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
		res, err := v.Search(ctx, Query{Text: "question", K: k, Channels: []string{"search-test"}})
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

// TestChannelAccess checks that both retrievers respect the channels of
// store.AllowedChannels. It needs migrations 003 and 004 applied, inserts one
// message per channel with ids from 93001 and, on cleanup, deletes only those
// ids. Results are narrowed to those ids so other rows do not matter.
func TestChannelAccess(t *testing.T) {
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

	channels := []string{"backend", "devops", "frontend", "product", "general"}
	idByChannel := map[string]int{}
	ids := make([]int, len(channels))
	for i, ch := range channels {
		ids[i] = accessBaseID + i
		idByChannel[ch] = ids[i]
	}
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
		msgs[i] = store.Message{ID: id, Channel: channels[i], Author: "test", Ts: ts, Text: "доступ каналов проверка"}
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

	// The query vector is equally close to every test message.
	qv := make([]float32, store.EmbeddingDim)
	for i := range ids {
		qv[i] = 1
	}
	retrievers := map[string]Retriever{
		"fts":    NewFTS(pool),
		"vector": NewVector(pool, fakeEmbedder{vec: qv}),
	}

	// k is large enough that rows already in the database cannot push the
	// test messages out of the result.
	const k = 1000
	idsOf := func(res []Result) map[int]bool {
		got := map[int]bool{}
		for _, r := range res {
			if r.ID >= accessBaseID && r.ID < accessBaseID+len(ids) {
				got[r.ID] = true
			}
		}
		return got
	}

	tests := []struct {
		user string
		want []string
	}{
		{"guest", []string{"general"}},
		{"nobody", nil},
		{"lead", channels},
	}
	for _, tt := range tests {
		allowed, err := store.AllowedChannels(ctx, pool, tt.user)
		if err != nil {
			t.Fatalf("AllowedChannels(%q): %v", tt.user, err)
		}
		for name, r := range retrievers {
			res, err := r.Search(ctx, Query{Text: "доступ каналов проверка", K: k, Channels: allowed})
			if err != nil {
				t.Fatalf("%s as %s: Search: %v", name, tt.user, err)
			}
			got := idsOf(res)
			want := map[int]bool{}
			for _, ch := range tt.want {
				want[idByChannel[ch]] = true
			}
			if len(got) != len(want) {
				t.Errorf("%s as %s: got ids %v, want %v", name, tt.user, got, want)
				continue
			}
			for id := range want {
				if !got[id] {
					t.Errorf("%s as %s: missing id %d, got %v", name, tt.user, id, got)
				}
			}
			if tt.user == "guest" && got[idByChannel["devops"]] {
				t.Errorf("%s as guest: got a devops message", name)
			}
		}
	}
}
