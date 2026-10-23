package search

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// FTS is a Retriever backed by PostgreSQL full-text search. The words of the
// query are combined with OR; Score is ts_rank (the larger, the better).
type FTS struct {
	pool *pgxpool.Pool
}

func NewFTS(pool *pgxpool.Pool) *FTS {
	return &FTS{pool: pool}
}

// plainto_tsquery joins words with & and never interprets operators in the
// user input; its text form quotes every lexeme, so replacing & with | only
// turns the AND between lexemes into OR.
const ftsSQL = `
WITH q AS (
    SELECT replace(plainto_tsquery('russian', $1)::text, '&', '|')::tsquery AS query
)
SELECT id, channel, author, ts, text, ts_rank(tsv, q.query)
FROM messages, q
WHERE tsv @@ q.query
ORDER BY ts_rank(tsv, q.query) DESC, id
LIMIT $2`

func (f *FTS) Search(ctx context.Context, q Query) ([]Result, error) {
	if q.K < 1 {
		return nil, fmt.Errorf("search: k must be at least 1, got %d", q.K)
	}

	rows, err := f.pool.Query(ctx, ftsSQL, q.Text, q.K)
	if err != nil {
		return nil, fmt.Errorf("query full-text messages: %w", err)
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
