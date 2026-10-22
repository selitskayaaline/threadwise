package rag

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/selitskayaaline/threadwise/internal/ollama"
	"github.com/selitskayaaline/threadwise/internal/search"
)

func ts(day int) time.Time { return time.Date(2026, 1, day, 10, 30, 0, 0, time.UTC) }

func sampleResults() []search.Result {
	return []search.Result{
		{ID: 7, Channel: "dev", Author: "Anna", TS: ts(5), Text: "позже"},
		{ID: 3, Channel: "ops", Author: "Boris", TS: ts(1), Text: "раньше"},
		{ID: 9, Channel: "dev", Author: "Clara", TS: ts(3), Text: "середина"},
	}
}

func TestBuildPrompt(t *testing.T) {
	msgs := BuildPrompt("когда релиз?", sampleResults())
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[1].Role != "user" {
		t.Fatalf("unexpected messages %+v", msgs)
	}
	user := msgs[1].Content

	start, end := strings.Index(user, "<messages>"), strings.Index(user, "</messages>")
	if start < 0 || end < start {
		t.Fatalf("delimiters missing or misordered in %q", user)
	}
	if !strings.Contains(user[end:], "когда релиз?") {
		t.Error("question must follow the closing delimiter")
	}

	block := user[start:end]
	pos := -1
	for _, line := range []string{
		"[3] #ops · Boris · 2026-01-01 10:30: раньше",
		"[9] #dev · Clara · 2026-01-03 10:30: середина",
		"[7] #dev · Anna · 2026-01-05 10:30: позже",
	} {
		i := strings.Index(block, line)
		if i < 0 {
			t.Fatalf("line %q not found in %q", line, block)
		}
		if i < pos {
			t.Errorf("line %q is out of chronological order", line)
		}
		pos = i
	}
}

func TestBuildPromptDoesNotReorderInput(t *testing.T) {
	in := sampleResults()
	BuildPrompt("q", in)
	if in[0].ID != 7 {
		t.Error("BuildPrompt modified its input")
	}
}

func TestSystemPromptMatchesDoc(t *testing.T) {
	doc, err := os.ReadFile("../../docs/prompt.md")
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(doc)) != systemPrompt {
		t.Error("systemPrompt differs from docs/prompt.md")
	}
}

func TestParseCitations(t *testing.T) {
	tests := []struct {
		name, in string
		want     []int
	}{
		{"single", "Релиз в пятницу [3].", []int{3}},
		{"list", "Так решили [3, 4].", []int{3, 4}},
		{"several and duplicates", "A [2], B [5], снова [2].", []int{2, 5}},
		{"none", "Ссылок нет.", nil},
		{"garbage", "Это [abc] и [] и [1a] и [-]", nil},
		{"garbage next to valid", "[abc] но [8]", []int{8}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseCitations(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

type fakeRetriever struct {
	res []search.Result
	err error
}

func (f fakeRetriever) Search(ctx context.Context, q string, k int) ([]search.Result, error) {
	return f.res, f.err
}

type fakeChat struct {
	reply string
	err   error
	calls int
	temp  float64
}

func (f *fakeChat) Chat(ctx context.Context, m []ollama.Message, temp float64) (string, error) {
	f.calls++
	f.temp = temp
	return f.reply, f.err
}

func TestAnswerCatchesInventedCitation(t *testing.T) {
	chat := &fakeChat{reply: "Релиз в пятницу [3], деплой в понедельник [42]."}
	s := &Service{Retriever: fakeRetriever{res: sampleResults()}, Chat: chat, K: 3, Temperature: 0.1}

	res, err := s.Answer(context.Background(), "когда релиз?")
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if !reflect.DeepEqual(res.Unknown, []int{42}) {
		t.Errorf("Unknown = %v, want [42]", res.Unknown)
	}
	if len(res.Cited) != 1 || res.Cited[0].ID != 3 {
		t.Errorf("Cited = %+v, want only message 3", res.Cited)
	}
	if chat.temp != 0.1 {
		t.Errorf("temperature = %v, want 0.1", chat.temp)
	}
}

func TestAnswerNoResultsSkipsChat(t *testing.T) {
	chat := &fakeChat{reply: "не должно быть вызвано"}
	s := &Service{Retriever: fakeRetriever{}, Chat: chat, K: 3}

	res, err := s.Answer(context.Background(), "q")
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if res.Text != NoAnswer || chat.calls != 0 {
		t.Errorf("got %q after %d chat calls", res.Text, chat.calls)
	}
}

func TestAnswerErrors(t *testing.T) {
	boom := errors.New("boom")
	s := &Service{Retriever: fakeRetriever{err: boom}, Chat: &fakeChat{}, K: 1}
	if _, err := s.Answer(context.Background(), "q"); !errors.Is(err, boom) {
		t.Errorf("search error: got %v", err)
	}
	s = &Service{Retriever: fakeRetriever{res: sampleResults()}, Chat: &fakeChat{err: boom}, K: 1}
	if _, err := s.Answer(context.Background(), "q"); !errors.Is(err, boom) {
		t.Errorf("chat error: got %v", err)
	}
}

func TestAnswerRefusalDropsCitations(t *testing.T) {
	tests := []struct{ name, reply string }{
		{"exact phrase after a claim", "[7] что-то. " + NoAnswer},
		{"lowercase without period", "в сообщениях нет ответа на этот вопрос"},
		{"lowercase after a claim", "[7] что-то. в сообщениях нет ответа на этот вопрос"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Service{Retriever: fakeRetriever{res: sampleResults()}, Chat: &fakeChat{reply: tt.reply}, K: 3}
			res, err := s.Answer(context.Background(), "q")
			if err != nil {
				t.Fatalf("Answer: %v", err)
			}
			if res.Text != NoAnswer {
				t.Errorf("Text = %q, want %q", res.Text, NoAnswer)
			}
			if len(res.Cited) != 0 || len(res.Unknown) != 0 {
				t.Errorf("Cited = %+v, Unknown = %v, want both empty", res.Cited, res.Unknown)
			}
		})
	}
}
