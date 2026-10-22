package store

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestUpsertEmbeddingInvalidation runs against a separate database identified
// by TEST_DATABASE_URL (never the dev database) and never deletes from
// messages. It does not assume anything about the row's prior state: step 2
// always overwrites embedding with a known value before the assertions.
func TestUpsertEmbeddingInvalidation(t *testing.T) {
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
	defer pool.Close()

	const id = 90001
	ts := time.Date(2026, 10, 5, 9, 0, 0, 0, time.UTC)

	msg := Message{ID: id, Channel: "general", Author: "test", Ts: ts, Text: "original"}
	if err := UpsertMessages(ctx, pool, []Message{msg}); err != nil {
		t.Fatalf("initial UpsertMessages: %v", err)
	}

	known := vectorLiteral(1024, 0.5)
	if _, err := pool.Exec(ctx, "UPDATE messages SET embedding = $1::vector WHERE id = $2", known, id); err != nil {
		t.Fatalf("set known embedding: %v", err)
	}

	// Same text: embedding must be preserved exactly.
	if err := UpsertMessages(ctx, pool, []Message{msg}); err != nil {
		t.Fatalf("UpsertMessages with unchanged text: %v", err)
	}
	if got := embeddingText(ctx, t, pool, id); got != known {
		t.Fatalf("embedding changed after unchanged-text upsert: got %q, want %q", got, known)
	}

	// Different text: embedding must be reset to NULL.
	msg.Text = "changed"
	if err := UpsertMessages(ctx, pool, []Message{msg}); err != nil {
		t.Fatalf("UpsertMessages with changed text: %v", err)
	}
	if got := embeddingIsNull(ctx, t, pool, id); !got {
		t.Fatalf("embedding not reset to NULL after text change")
	}
}

func embeddingText(ctx context.Context, t *testing.T, pool *pgxpool.Pool, id int) string {
	t.Helper()
	var text *string
	if err := pool.QueryRow(ctx, "SELECT embedding::text FROM messages WHERE id = $1", id).Scan(&text); err != nil {
		t.Fatalf("read embedding: %v", err)
	}
	if text == nil {
		return ""
	}
	return *text
}

func embeddingIsNull(ctx context.Context, t *testing.T, pool *pgxpool.Pool, id int) bool {
	t.Helper()
	var isNull bool
	if err := pool.QueryRow(ctx, "SELECT embedding IS NULL FROM messages WHERE id = $1", id).Scan(&isNull); err != nil {
		t.Fatalf("check embedding null: %v", err)
	}
	return isNull
}

func vectorLiteral(n int, v float64) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = strconv.FormatFloat(v, 'f', 1, 64)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
