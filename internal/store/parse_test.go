package store

import (
	"os"
	"testing"
	"time"
)

func TestParseMessages(t *testing.T) {
	data, err := os.ReadFile("testdata/messages.json")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}

	msgs, err := ParseMessages(data)
	if err != nil {
		t.Fatalf("ParseMessages: %v", err)
	}

	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}

	got := msgs[0]
	wantTs := time.Date(2026, 10, 5, 9, 0, 0, 0, time.UTC)

	if got.ID != 1 {
		t.Errorf("ID = %d, want 1", got.ID)
	}
	if got.Channel != "general" {
		t.Errorf("Channel = %q, want %q", got.Channel, "general")
	}
	if got.Author != "Вика" {
		t.Errorf("Author = %q, want %q", got.Author, "Вика")
	}
	if !got.Ts.Equal(wantTs) {
		t.Errorf("Ts = %v, want %v", got.Ts, wantTs)
	}
	if got.Ts.Location() != time.UTC {
		t.Errorf("Ts location = %v, want UTC", got.Ts.Location())
	}
	if got.Text != "С понедельником всех, го в бой" {
		t.Errorf("Text = %q, want %q", got.Text, "С понедельником всех, го в бой")
	}
}
