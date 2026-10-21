-include .env
export

test:
	go test ./...

load:
	go run ./cmd/threadwise load
