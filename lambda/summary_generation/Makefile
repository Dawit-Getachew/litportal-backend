# Lambda boilerplate Makefile
# Build and test commands for local development

LAMBDA_NAME := litportal-summary-generation
DOCKER_IMAGE := $(LAMBDA_NAME)
PYTHON_VERSION := 3.12
PORT := 9000
EVENT_FILE := events/sqs_event.json


.PHONY: build
build:
	@echo "Building Docker image: $(DOCKER_IMAGE)..."
	@cd ../.. && docker build --platform linux/amd64 -f lambda/summary_generation/Dockerfile -t $(DOCKER_IMAGE) .
	@echo "Docker image built successfully"

.PHONY: run
run: build
	@echo "Starting Lambda container on port $(PORT)..."
	@docker stop $(LAMBDA_NAME) 2>/dev/null || true
	@docker run -d --rm --name $(LAMBDA_NAME) -p $(PORT):8080 $(DOCKER_IMAGE) > /dev/null
	@echo "Container started. Send events with: make send-event"
	@echo "View logs with:                      make logs"
	@echo "Stop with:                           make stop"

.PHONY: stop
stop:
	@docker stop $(LAMBDA_NAME) 2>/dev/null && echo "Container stopped" || echo "Container not running"

.PHONY: logs
logs:
	@docker logs -f $(LAMBDA_NAME)

.PHONY: send-event
send-event:
	@echo "Sending event from $(EVENT_FILE)..."
	@curl -s -X POST "http://localhost:$(PORT)/2015-03-31/functions/function/invocations" \
		-H "Content-Type: application/json" \
		-d @$(EVENT_FILE) | python3 -m json.tool

.PHONY: test
test:
	@uv run pytest tests/ -v

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@docker stop $(LAMBDA_NAME) 2>/dev/null || true
	@docker rmi $(DOCKER_IMAGE) 2>/dev/null || true
	@rm -rf __pycache__ .pytest_cache .coverage src/__pycache__
	@echo "Clean completed"

# Builds, runs, invokes, and tears down — no manual steps needed
.PHONY: test-container
test-container: build
	@echo "Testing lambda in container..."
	@docker stop $(LAMBDA_NAME)-test 2>/dev/null || true
	@docker rm $(LAMBDA_NAME)-test 2>/dev/null || true
	@echo "Starting container..."
	@docker run -d --name $(LAMBDA_NAME)-test -p $(PORT):8080 $(DOCKER_IMAGE) > /dev/null
	@echo "Waiting for container to be ready..."
	@until curl -sf -o /dev/null -X POST "http://localhost:$(PORT)/2015-03-31/functions/function/invocations" \
		-H "Content-Type: application/json" -d '{}' 2>/dev/null; do sleep 0.5; done
	@echo "Sending event from $(EVENT_FILE)..."
	@curl -s -X POST "http://localhost:$(PORT)/2015-03-31/functions/function/invocations" \
		-H "Content-Type: application/json" \
		-d @$(EVENT_FILE) | python3 -m json.tool
	@docker stop $(LAMBDA_NAME)-test > /dev/null
	@docker rm $(LAMBDA_NAME)-test > /dev/null
	@echo "Test completed"
