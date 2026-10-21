package store

import (
	"context"
	"encoding/json"
	"fmt"
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

	const stmt = `
		INSERT INTO messages (id, channel, author, ts, text)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (id) DO UPDATE SET
			channel = excluded.channel,
			author  = excluded.author,
			ts      = excluded.ts,
			text    = excluded.text
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
