# Makefile for Stock Sentiment Analysis System
# Automatically cleans volumes before building

.PHONY: build clean-volumes up down rebuild logs shell test-all help

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BLUE)Stock Sentiment Analysis System - Docker Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

clean-volumes: ## Clean host volume directories
	@echo "$(BLUE)[INFO]$(NC) Cleaning host volumes..."
	@mkdir -p container_output container_logs
	@rm -rf container_output/* container_logs/* 2>/dev/null || true
	@mkdir -p container_output/news container_output/tradingview container_output/images container_output/final_score container_output/charts
	@echo "$(GREEN)[SUCCESS]$(NC) Host volumes cleaned"

build: clean-volumes ## Clean volumes and build container
	@echo "$(BLUE)[INFO]$(NC) Building container with clean volumes..."
	@docker-compose build --no-cache
	@echo "$(GREEN)[SUCCESS]$(NC) Container built successfully"

up: clean-volumes ## Clean volumes and start container
	@echo "$(BLUE)[INFO]$(NC) Starting container with clean volumes..."
	@docker-compose up --build -d
	@echo "$(GREEN)[SUCCESS]$(NC) Container started"
	@$(MAKE) status

down: ## Stop and remove container
	@echo "$(BLUE)[INFO]$(NC) Stopping container..."
	@docker-compose down
	@echo "$(GREEN)[SUCCESS]$(NC) Container stopped"

rebuild: down build up ## Complete rebuild: stop, clean, build, start

status: ## Show container status
	@echo "$(BLUE)[INFO]$(NC) Container status:"
	@docker ps -a --filter name=stock-sentiment-app --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true

logs: ## View container logs
	@docker-compose logs -f app

shell: ## Access container shell
	@docker-compose exec app shell

test-all: ## Run comprehensive test suite
	@docker-compose exec app test-all

automated: ## Run automated sentiment analysis
	@docker-compose exec app automated

news: ## Run news analysis only
	@docker-compose exec app news

images: ## Run image analysis only
	@docker-compose exec app images

tradingview: ## Run TradingView analysis only
	@docker-compose exec app tradingview

quick-test: ## Run quick system health check
	@docker-compose exec app test

clean-all: down ## Stop container and clean everything
	@$(MAKE) clean-volumes
	@docker system prune -f 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) Everything cleaned"

# Verify volumes are clean
verify-clean: ## Verify that volumes are clean
	@echo "$(BLUE)[INFO]$(NC) Verifying clean state..."
	@echo "Host volumes:"
	@ls -la container_output/ 2>/dev/null || echo "  container_output/: (empty)"
	@ls -la container_logs/ 2>/dev/null || echo "  container_logs/: (empty)"
	@echo "Container volumes:"
	@docker-compose exec app ls -la /app/Output/ 2>/dev/null || echo "  /app/Output/: (container not running)"
	@docker-compose exec app ls -la /app/logs/ 2>/dev/null || echo "  /app/logs/: (container not running)"