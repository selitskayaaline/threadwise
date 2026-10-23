package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/selitskayaaline/threadwise/internal/search"
)

// Question is a test question with the ids of the messages that answer it.
type Question struct {
	ID             int    `json:"id"`
	Question       string `json:"question"`
	GoldIDs        []int  `json:"gold_ids"`
	ExpectedAnswer string `json:"expected_answer"`
}

// Metrics are the retrieval metrics of one method over N questions.
type Metrics struct {
	Hit float64
	MRR float64
	N   int
}

// Miss is a question for which no gold message was found in the top k.
type Miss struct {
	QuestionID int
	Question   string
	GoldIDs    []int
	Found      []int
}

// MethodResult is the outcome of running one method over all questions.
type MethodResult struct {
	Method  string
	Metrics Metrics
	Misses  []Miss
}

// LoadQuestions parses the JSON list of questions.
func LoadQuestions(data []byte) ([]Question, error) {
	var qs []Question
	if err := json.Unmarshal(data, &qs); err != nil {
		return nil, fmt.Errorf("parse questions: %w", err)
	}
	return qs, nil
}

// SplitAnswerable drops the questions without gold_ids and reports how many
// were dropped.
func SplitAnswerable(qs []Question) (use []Question, skipped int) {
	for _, q := range qs {
		if len(q.GoldIDs) == 0 {
			skipped++
			continue
		}
		use = append(use, q)
	}
	return use, skipped
}

// Score checks the first k found ids against gold. rr is 1/rank of the first
// gold id (rank starts at 1), or 0 on a miss.
func Score(found, gold []int, k int) (hit bool, rr float64) {
	isGold := make(map[int]bool, len(gold))
	for _, id := range gold {
		isGold[id] = true
	}
	if k > len(found) {
		k = len(found)
	}
	for i, id := range found[:k] {
		if isGold[id] {
			return true, 1 / float64(i+1)
		}
	}
	return false, 0
}

// Run searches every question with r among channels and computes hit@k and MRR@k.
func Run(ctx context.Context, r search.Retriever, qs []Question, k int, channels []string) (Metrics, []Miss, error) {
	var m Metrics
	var misses []Miss
	hits := 0
	sumRR := 0.0
	for _, q := range qs {
		res, err := r.Search(ctx, search.Query{Text: q.Question, K: k, Channels: channels})
		if err != nil {
			return Metrics{}, nil, fmt.Errorf("question %d: %w", q.ID, err)
		}
		found := make([]int, len(res))
		for i, x := range res {
			found[i] = x.ID
		}
		hit, rr := Score(found, q.GoldIDs, k)
		if hit {
			hits++
		} else {
			misses = append(misses, Miss{QuestionID: q.ID, Question: q.Question, GoldIDs: q.GoldIDs, Found: found})
		}
		sumRR += rr
	}
	m.N = len(qs)
	if m.N > 0 {
		m.Hit = float64(hits) / float64(m.N)
		m.MRR = sumRR / float64(m.N)
	}
	return m, misses, nil
}

// WriteReport prints a Markdown table of the methods followed by the misses
// of each method.
func WriteReport(w io.Writer, k, skipped int, results []MethodResult) {
	fmt.Fprintf(w, "Пропущено вопросов без gold_ids: %d\n\n", skipped)
	fmt.Fprintf(w, "| метод | hit@%d | MRR@%d | число вопросов |\n", k, k)
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range results {
		fmt.Fprintf(w, "| %s | %.3f | %.3f | %d |\n", r.Method, r.Metrics.Hit, r.Metrics.MRR, r.Metrics.N)
	}
	for _, r := range results {
		fmt.Fprintf(w, "\n## Промахи: %s (%d)\n", r.Method, len(r.Misses))
		for _, m := range r.Misses {
			fmt.Fprintf(w, "- #%d %s\n  gold_ids: %s; найдено: %s\n", m.QuestionID, m.Question, formatIDs(m.GoldIDs), formatIDs(m.Found))
		}
	}
}

func formatIDs(ids []int) string {
	if len(ids) == 0 {
		return "[]"
	}
	s := make([]string, len(ids))
	for i, id := range ids {
		s[i] = fmt.Sprint(id)
	}
	return "[" + strings.Join(s, ", ") + "]"
}
