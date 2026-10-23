package store

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Message struct {
	ID      int
	Channel string
	Author  string
	Ts      time.Time
	Text    string
}

type rawMessage struct {
	ID      int    `json:"id"`
	Channel string `json:"channel"`
	Author  string `json:"author"`
	Ts      string `json:"ts"`
	Text    string `json:"text"`
}

// tsLayout matches data/messages.json, where timestamps carry no UTC offset
// (e.g. "2026-10-05T09:00:00"). time.Time cannot unmarshal such a value
// directly, so messages are decoded via rawMessage and the timestamp is parsed
// explicitly here, treating it as UTC to match the TIMESTAMPTZ column.
const tsLayout = "2006-01-02T15:04:05"

func ParseMessages(data []byte) ([]Message, error) {
	var raw []rawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse messages json: %w", err)
	}

	msgs := make([]Message, 0, len(raw))
	for _, r := range raw {
		ts, err := time.ParseInLocation(tsLayout, r.Ts, time.UTC)
		if err != nil {
			return nil, fmt.Errorf("parse ts for message id=%d: %w", r.ID, err)
		}
		msgs = append(msgs, Message{
			ID:      r.ID,
			Channel: r.Channel,
			Author:  r.Author,
			Ts:      ts,
			Text:    r.Text,
		})
	}
	return msgs, nil
}

func UpsertMessages(ctx context.Context, pool *pgxpool.Pool, msgs []Message) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	// embedding is derived from text, so a text change invalidates it; an
	// unchanged text keeps the previously computed embedding.
	const stmt = `
		INSERT INTO messages (id, channel, author, ts, text)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (id) DO UPDATE SET
			channel   = excluded.channel,
			author    = excluded.author,
			ts        = excluded.ts,
			text      = excluded.text,
			embedding = CASE
				WHEN messages.text IS DISTINCT FROM excluded.text THEN NULL
				ELSE messages.embedding
			END
	`
	for _, m := range msgs {
		if _, err := tx.Exec(ctx, stmt, m.ID, m.Channel, m.Author, m.Ts, m.Text); err != nil {
			return fmt.Errorf("upsert message id=%d: %w", m.ID, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

// EmbeddingDim is the dimension of the messages.embedding column.
const EmbeddingDim = 1024

// FormatVector renders v in the pgvector text format, e.g. "[0.1,0.2]".
func FormatVector(v []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, x := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(x), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

func checkDim(id int, v []float32) error {
	if len(v) != EmbeddingDim {
		return fmt.Errorf("message id=%d: модель вернула %d, колонка ждёт %d", id, len(v), EmbeddingDim)
	}
	return nil
}

// PendingMessages returns up to limit messages that have no embedding yet.
func PendingMessages(ctx context.Context, pool *pgxpool.Pool, limit int) ([]Message, error) {
	rows, err := pool.Query(ctx, `
		SELECT id, channel, author, ts, text
		FROM messages
		WHERE embedding IS NULL
		ORDER BY id
		LIMIT $1
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("query messages without embedding: %w", err)
	}
	defer rows.Close()

	var msgs []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.Channel, &m.Author, &m.Ts, &m.Text); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		msgs = append(msgs, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read messages: %w", err)
	}
	return msgs, nil
}

// SetEmbeddings stores vecs[i] for msgs[i] and returns how many rows were
// updated. Every vector is validated before anything is written. A row is
// skipped if its text changed since msgs was read (the vector would be stale),
// so it stays NULL and is picked up by the next run.
func SetEmbeddings(ctx context.Context, pool *pgxpool.Pool, msgs []Message, vecs [][]float32) (int, error) {
	if len(msgs) != len(vecs) {
		return 0, fmt.Errorf("got %d vectors for %d messages", len(vecs), len(msgs))
	}
	for i, m := range msgs {
		if err := checkDim(m.ID, vecs[i]); err != nil {
			return 0, err
		}
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	const stmt = `
		UPDATE messages SET embedding = $1::vector
		WHERE id = $2 AND text = $3 AND embedding IS NULL
	`
	updated := 0
	for i, m := range msgs {
		tag, err := tx.Exec(ctx, stmt, FormatVector(vecs[i]), m.ID, m.Text)
		if err != nil {
			return 0, fmt.Errorf("set embedding for message id=%d: %w", m.ID, err)
		}
		updated += int(tag.RowsAffected())
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit transaction: %w", err)
	}
	return updated, nil
}

// AllowedChannels returns the channels the user may search, sorted by name.
// An unknown user has no channels; that is an empty result, not an error.
func AllowedChannels(ctx context.Context, pool *pgxpool.Pool, user string) ([]string, error) {
	rows, err := pool.Query(ctx, `
		SELECT channel
		FROM channel_access
		WHERE user_name = $1
		ORDER BY channel
	`, user)
	if err != nil {
		return nil, fmt.Errorf("query channels of user %q: %w", user, err)
	}
	defer rows.Close()

	channels := []string{}
	for rows.Next() {
		var c string
		if err := rows.Scan(&c); err != nil {
			return nil, fmt.Errorf("scan channel: %w", err)
		}
		channels = append(channels, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read channels: %w", err)
	}
	return channels, nil
}
