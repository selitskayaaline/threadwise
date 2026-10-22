-include .env
export

test:
	go test ./...

load:
	go run ./cmd/threadwise load

embed:
	go run ./cmd/threadwise embed
