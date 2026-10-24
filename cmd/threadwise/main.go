package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/selitskayaaline/threadwise/internal/config"
	"github.com/selitskayaaline/threadwise/internal/eval"
	"github.com/selitskayaaline/threadwise/internal/ollama"
	"github.com/selitskayaaline/threadwise/internal/rag"
	"github.com/selitskayaaline/threadwise/internal/search"
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
	case "search":
		cmdSearch(os.Args[2:])
	case "ask":
		cmdAsk(os.Args[2:])
	case "eval":
		cmdEval(os.Args[2:])
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
	fmt.Fprintln(os.Stderr, "  search  find the messages closest to a question")
	fmt.Fprintln(os.Stderr, "  ask     answer a question from the found messages, with sources")
	fmt.Fprintln(os.Stderr, "  eval    compare search methods on the test questions (hit@k, MRR)")
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

func cmdSearch(args []string) {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	k := fs.Int("k", 5, "number of messages to return")
	method := fs.String("method", "vector", "search method: vector or fts")
	user := fs.String("user", defaultUser, "user whose channels are searched")
	fs.Parse(args)

	if fs.NArg() != 1 || fs.Arg(0) == "" {
		fmt.Fprintln(os.Stderr, "usage: threadwise search [-k N] [-method vector|fts] [-user NAME] \"question\"")
		os.Exit(2)
	}
	if *k < 1 {
		fmt.Fprintln(os.Stderr, "threadwise search: -k must be at least 1")
		os.Exit(1)
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise search:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise search:", err)
		os.Exit(1)
	}
	defer pool.Close()

	var retriever search.Retriever
	switch *method {
	case "fts":
		retriever = search.NewFTS(pool)
	case "vector":
		if err := cfg.RequireOllama(); err != nil {
			fmt.Fprintln(os.Stderr, "threadwise search:", err)
			os.Exit(1)
		}
		retriever = search.NewVector(pool, ollama.New(cfg.OllamaURL, cfg.EmbedModel))
	default:
		fmt.Fprintf(os.Stderr, "threadwise search: unknown -method %q (want vector or fts)\n", *method)
		os.Exit(2)
	}
	channels, err := store.AllowedChannels(ctx, pool, *user)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise search:", err)
		os.Exit(1)
	}
	results, err := retriever.Search(ctx, search.Query{Text: fs.Arg(0), K: *k, Channels: channels})
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise search:", err)
		os.Exit(1)
	}

	for _, r := range results {
		fmt.Printf("%d\t%s\t%s\t%s\t%.3f\t%s\n",
			r.ID, r.Channel, r.Author, r.TS.UTC().Format(time.RFC3339), r.Score, r.Text)
	}
}

const askTemperature = 0.1

// defaultUser is the user search and ask run as without -user; evalUser is the
// user eval runs as.
const (
	defaultUser = "lead"
	evalUser    = "lead"
)

func cmdAsk(args []string) {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	k := fs.Int("k", 5, "number of messages to give the model")
	method := fs.String("method", "vector", "search method: vector or fts")
	user := fs.String("user", defaultUser, "user whose channels are searched")
	fs.Parse(args)

	if fs.NArg() != 1 || fs.Arg(0) == "" {
		fmt.Fprintln(os.Stderr, "usage: threadwise ask [-k N] [-method vector|fts] [-user NAME] \"question\"")
		os.Exit(2)
	}
	if *k < 1 {
		fmt.Fprintln(os.Stderr, "threadwise ask: -k must be at least 1")
		os.Exit(1)
	}
	if *method != "vector" && *method != "fts" {
		fmt.Fprintf(os.Stderr, "threadwise ask: unknown -method %q (want vector or fts)\n", *method)
		os.Exit(2)
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise ask:", err)
		os.Exit(1)
	}
	if err := cfg.RequireChat(); err != nil {
		fmt.Fprintln(os.Stderr, "threadwise ask:", err)
		os.Exit(1)
	}
	if *method == "vector" {
		if err := cfg.RequireOllama(); err != nil {
			fmt.Fprintln(os.Stderr, "threadwise ask:", err)
			os.Exit(1)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise ask:", err)
		os.Exit(1)
	}
	defer pool.Close()

	var retriever search.Retriever
	if *method == "fts" {
		retriever = search.NewFTS(pool)
	} else {
		retriever = search.NewVector(pool, ollama.New(cfg.OllamaURL, cfg.EmbedModel))
	}

	channels, err := store.AllowedChannels(ctx, pool, *user)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise ask:", err)
		os.Exit(1)
	}

	svc := &rag.Service{
		Retriever:   retriever,
		Chat:        ollama.New(cfg.OllamaURL, cfg.ChatModel),
		K:           *k,
		Channels:    channels,
		Temperature: askTemperature,
	}
	res, err := svc.Answer(ctx, fs.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise ask:", err)
		os.Exit(1)
	}

	fmt.Println(res.Text)
	if len(res.Cited) > 0 {
		fmt.Println()
		fmt.Println("Источники:")
		for _, r := range res.Cited {
			fmt.Printf("[%d] %s, %s\n", r.ID, r.Author, rag.FormatTime(r))
		}
	}
	for _, id := range res.Unknown {
		fmt.Fprintf(os.Stderr, "предупреждение: модель сослалась на несуществующее сообщение [%d]\n", id)
	}
}

func cmdEval(args []string) {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	k := fs.Int("k", 5, "number of messages to retrieve per question")
	methods := fs.String("methods", "fts,vector", "comma-separated search methods: fts, vector")
	file := fs.String("file", "data/questions.json", "path to the questions JSON file")
	attacks := fs.Bool("attacks", false, "measure prompt-injection attacks with "+attackQuestionsFile+" instead of retrieval quality")
	fs.Parse(args)

	if *k < 1 {
		fmt.Fprintln(os.Stderr, "threadwise eval: -k must be at least 1")
		os.Exit(1)
	}

	var names []string
	seen := map[string]bool{}
	needVector := false
	for _, name := range strings.Split(*methods, ",") {
		name = strings.TrimSpace(name)
		if name != "fts" && name != "vector" {
			fmt.Fprintf(os.Stderr, "threadwise eval: unknown method %q in -methods (want fts or vector)\n", name)
			os.Exit(2)
		}
		if seen[name] {
			fmt.Fprintf(os.Stderr, "threadwise eval: method %q is listed twice in -methods\n", name)
			os.Exit(2)
		}
		seen[name] = true
		names = append(names, name)
		needVector = needVector || name == "vector"
	}

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	if needVector {
		if err := cfg.RequireOllama(); err != nil {
			fmt.Fprintln(os.Stderr, "threadwise eval:", err)
			os.Exit(1)
		}
	}

	if *attacks {
		if err := cfg.RequireChat(); err != nil {
			fmt.Fprintln(os.Stderr, "threadwise eval:", err)
			os.Exit(1)
		}
		cmdEvalAttacks(cfg, names, *k)
		return
	}

	data, err := os.ReadFile(*file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	all, err := eval.LoadQuestions(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	qs, skipped := eval.SplitAnswerable(all)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	defer pool.Close()

	channels, err := store.AllowedChannels(ctx, pool, evalUser)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}

	var results []eval.MethodResult
	for _, name := range names {
		var retriever search.Retriever
		if name == "fts" {
			retriever = search.NewFTS(pool)
		} else {
			retriever = search.NewVector(pool, ollama.New(cfg.OllamaURL, cfg.EmbedModel))
		}
		m, misses, err := eval.Run(ctx, retriever, qs, *k, channels)
		if err != nil {
			fmt.Fprintf(os.Stderr, "threadwise eval: method %s: %v\n", name, err)
			os.Exit(1)
		}
		results = append(results, eval.MethodResult{Method: name, Metrics: m, Misses: misses})
	}

	eval.WriteReport(os.Stdout, *k, skipped, results)
}

const attackQuestionsFile = "data/attack_questions.json"

// cmdEvalAttacks asks the attack questions through rag.Service, as ask does,
// for every method and prints the share of successful attacks and the answers.
func cmdEvalAttacks(cfg config.Config, methods []string, k int) {
	data, err := os.ReadFile(attackQuestionsFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	qs, err := eval.LoadAttackQuestions(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}
	defer pool.Close()

	channels, err := store.AllowedChannels(ctx, pool, evalUser)
	if err != nil {
		fmt.Fprintln(os.Stderr, "threadwise eval:", err)
		os.Exit(1)
	}

	var results []eval.AttackResult
	for _, name := range methods {
		var retriever search.Retriever
		if name == "fts" {
			retriever = search.NewFTS(pool)
		} else {
			retriever = search.NewVector(pool, ollama.New(cfg.OllamaURL, cfg.EmbedModel))
		}
		svc := &rag.Service{
			Retriever:   retriever,
			Chat:        ollama.New(cfg.OllamaURL, cfg.ChatModel),
			K:           k,
			Channels:    channels,
			Temperature: askTemperature,
		}

		res := eval.AttackResult{Method: name}
		for _, q := range qs {
			// A separate search with the same parameters shows whether the
			// attack reached the model; it does not affect the answer.
			found, err := retriever.Search(ctx, search.Query{Text: q.Question, K: k, Channels: channels})
			if err != nil {
				fmt.Fprintf(os.Stderr, "threadwise eval: method %s: question %d: %v\n", name, q.ID, err)
				os.Exit(1)
			}
			ids := make([]int, len(found))
			for i, f := range found {
				ids[i] = f.ID
			}

			ans, err := svc.Answer(ctx, q.Question)
			if err != nil {
				fmt.Fprintf(os.Stderr, "threadwise eval: method %s: question %d: %v\n", name, q.ID, err)
				os.Exit(1)
			}
			res.Outcomes = append(res.Outcomes, eval.AttackOutcome{
				Question: q,
				Answer:   ans.Text,
				Success:  eval.AttackSucceeded(ans.Text),
				InTopK:   eval.AttackIDs(ids),
			})
		}
		results = append(results, res)
	}

	eval.WriteAttackReport(os.Stdout, results)
}
