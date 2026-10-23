package store

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestUpsertMessagesIdempotent runs against a separate database identified by
// TEST_DATABASE_URL (never the dev database). It skips entirely when that
// variable is unset. The test only inserts/updates rows and reads row counts —
// it never deletes from messages.
func TestUpsertMessagesIdempotent(t *testing.T) {
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

	msgs := []Message{
		{ID: 1, Channel: "general", Author: "Вика", Ts: time.Date(2026, 10, 5, 9, 0, 0, 0, time.UTC), Text: "hello"},
		{ID: 2, Channel: "devops", Author: "Ира", Ts: time.Date(2026, 10, 5, 10, 15, 0, 0, time.UTC), Text: "world"},
	}

	if err := UpsertMessages(ctx, pool, msgs); err != nil {
		t.Fatalf("first UpsertMessages: %v", err)
	}
	first := countRows(ctx, t, pool)

	if err := UpsertMessages(ctx, pool, msgs); err != nil {
		t.Fatalf("second UpsertMessages: %v", err)
	}
	second := countRows(ctx, t, pool)

	if first != second {
		t.Fatalf("row count changed after repeat upsert: first=%d second=%d", first, second)
	}
}

func countRows(ctx context.Context, t *testing.T, pool *pgxpool.Pool) int {
	t.Helper()
	var count int
	if err := pool.QueryRow(ctx, "SELECT COUNT(*) FROM messages").Scan(&count); err != nil {
		t.Fatalf("count rows: %v", err)
	}
	return count
}

// TestAllowedChannels reads the channel_access rows from migration 004; it
// writes nothing.
func TestAllowedChannels(t *testing.T) {
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

	tests := []struct {
		user string
		want []string
	}{
		{"lead", []string{"backend", "devops", "frontend", "general", "product"}},
		{"frontend-dev", []string{"frontend", "general", "product"}},
		{"guest", []string{"general"}},
		{"nobody", []string{}},
	}
	for _, tt := range tests {
		got, err := AllowedChannels(ctx, pool, tt.user)
		if err != nil {
			t.Fatalf("AllowedChannels(%q): %v", tt.user, err)
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("AllowedChannels(%q) = %v, want %v", tt.user, got, tt.want)
		}
	}
}
