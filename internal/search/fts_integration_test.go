package search

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/store"
)

const ftsBaseID = 92001

// TestFTSSearch runs against the database from TEST_DATABASE_URL (never the
// dev database) and needs migration 003 applied. It inserts three messages
// with ids from 92001 and, on cleanup, deletes only those ids.
func TestFTSSearch(t *testing.T) {
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

	ids := []int{ftsBaseID, ftsBaseID + 1, ftsBaseID + 2}
	cleanup := func() {
		if _, err := pool.Exec(context.Background(), "DELETE FROM messages WHERE id = ANY($1)", ids); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	}
	cleanup()
	t.Cleanup(cleanup)

	ts := time.Date(2026, 10, 5, 9, 0, 0, 0, time.UTC)
	texts := []string{
		"Релиз переносим на пятницу",
		"Обед сегодня в столовой",
		"Погода сегодня хорошая",
	}
	msgs := make([]store.Message, len(ids))
	for i, id := range ids {
		msgs[i] = store.Message{ID: id, Channel: "fts-test", Author: "test", Ts: ts, Text: texts[i]}
	}
	if err := store.UpsertMessages(ctx, pool, msgs); err != nil {
		t.Fatalf("UpsertMessages: %v", err)
	}

	f := NewFTS(pool)

	// idsOf keeps only the test messages, so rows already in the database
	// do not affect the checks.
	idsOf := func(res []Result) map[int]bool {
		got := map[int]bool{}
		for _, r := range res {
			if r.ID >= ftsBaseID && r.ID < ftsBaseID+len(ids) {
				got[r.ID] = true
			}
		}
		return got
	}

	t.Run("other word form", func(t *testing.T) {
		res, err := f.Search(ctx, "готов ли план по релизу", 50)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if got := idsOf(res); !got[ftsBaseID] {
			t.Errorf("query with \"релизу\" did not find the message with \"релиз\"; got %v", got)
		}
	})

	t.Run("words are combined with OR", func(t *testing.T) {
		res, err := f.Search(ctx, "релизу обед", 50)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		got := idsOf(res)
		if !got[ftsBaseID] || !got[ftsBaseID+1] {
			t.Errorf("want messages %d and %d, got %v", ftsBaseID, ftsBaseID+1, got)
		}
		if got[ftsBaseID+2] {
			t.Errorf("message %d has none of the words but was found", ftsBaseID+2)
		}
		for i := 1; i < len(res); i++ {
			if res[i].Score > res[i-1].Score {
				t.Errorf("results not ordered by rank")
			}
		}
	})

	t.Run("limit", func(t *testing.T) {
		res, err := f.Search(ctx, "релизу обед сегодня", 1)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(res) != 1 {
			t.Errorf("k=1: got %d results", len(res))
		}
	})

	t.Run("stop words only", func(t *testing.T) {
		res, err := f.Search(ctx, "и в на", 5)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(res) != 0 {
			t.Errorf("got %d results for a query of stop words", len(res))
		}
	})

	t.Run("invalid k", func(t *testing.T) {
		if _, err := f.Search(ctx, "релиз", 0); err == nil {
			t.Error("k=0: expected an error")
		}
	})
}
