package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/config"
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
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: threadwise <command> [flags]")
	fmt.Fprintln(os.Stderr, "commands:")
	fmt.Fprintln(os.Stderr, "  load    load messages from a JSON file into the database")
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
