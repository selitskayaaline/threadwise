package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/config"
	"github.com/selitskayaaline/threadwise/internal/ollama"
	"github.com/selitskayaaline/threadwise/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	switch os.Args[1] {
	case "load":
		cmdLoad(os.Args[2:])
	case "embed":
		cmdEmbed(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: threadwise <command> [flags]")
	fmt.Fprintln(os.Stderr, "commands:")
	fmt.Fprintln(os.Stderr, "  load    load messages from a JSON file into the database")
	fmt.Fprintln(os.Stderr, "  embed   compute embeddings for messages that have none")
}

func cmdLoad(args []string) {
	fs := flag.NewFlagSet("load", flag.ExitOnError)
	file := fs.String("file", "data/messages.json", "path to the messages JSON file")
	fs.Parse(args)

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise load:", err)
		os.Exit(1)
	}

	data, err := os.ReadFile(*file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise load:", err)
		os.Exit(1)
	}

	msgs, err := store.ParseMessages(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise load:", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise load:", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := store.UpsertMessages(ctx, pool, msgs); err != nil {
		fmt.Fprintln(os.Stderr, "threadwise load:", err)
		os.Exit(1)
	}

	fmt.Printf("loaded %d messages\n", len(msgs))
}

func cmdEmbed(args []string) {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	batch := fs.Int("batch", 16, "number of messages per Ollama request")
	fs.Parse(args)

	if *batch < 1 {
		fmt.Fprintln(os.Stderr, "threadwise embed: -batch must be at least 1")
		os.Exit(1)
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise embed:", err)
		os.Exit(1)
	}
	if err := cfg.RequireOllama(); err != nil {
		fmt.Fprintln(os.Stderr, "threadwise embed:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise embed:", err)
		os.Exit(1)
	}
	defer pool.Close()

	client := ollama.New(cfg.OllamaURL, cfg.EmbedModel)

	total := 0
	for {
		msgs, err := store.PendingMessages(ctx, pool, *batch)
		if err != nil {
			fmt.Fprintln(os.Stderr, "threadwise embed:", err)
			os.Exit(1)
		}
		if len(msgs) == 0 {
			break
		}

		texts := make([]string, len(msgs))
		for i, m := range msgs {
			texts[i] = m.Text
		}
		vecs, err := client.Embed(ctx, texts)
		if err != nil {
			fmt.Fprintln(os.Stderr, "threadwise embed:", err)
			os.Exit(1)
		}

		n, err := store.SetEmbeddings(ctx, pool, msgs, vecs)
		if err != nil {
			fmt.Fprintln(os.Stderr, "threadwise embed:", err)
			os.Exit(1)
		}
		if n == 0 {
			fmt.Fprintln(os.Stderr, "threadwise embed: no rows updated for a batch, stopping")
			os.Exit(1)
		}
		total += n
	}

	fmt.Printf("embedded %d messages\n", total)
}
