-include .env
export

test:
	go test ./...

load:
	go run ./cmd/threadwise load

embed:
	go run ./cmd/threadwise embed

test-db:
	for f in migrations/*.sql; do \
		docker exec -i pgv psql -U postgres -d threadwise_test -v ON_ERROR_STOP=1 < $$f || exit 1; \
	done

eval:
	go run ./cmd/threadwise eval
