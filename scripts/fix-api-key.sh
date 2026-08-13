#!/bin/bash

# Script to validate configured provider credentials and restart services

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"
ENV_FILE="$ROOT_DIR/.env"
if docker compose version >/dev/null 2>&1; then
    COMPOSE_BIN=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_BIN=(docker-compose)
else
    echo "Docker Compose v2 or docker-compose is required to restart services."
    exit 1
fi

echo "🔧 KineGraph - Provider Key Setup"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking configured provider credentials...${NC}"

echo -e "${YELLOW}📋 Steps to Fix:${NC}"
echo ""
echo "1. Configure one supported provider key in .env:"
echo "   OPENAI_API_KEY, OPENROUTER_API_KEY, FIREWORKS_API_KEY, or NVIDIA_API_KEY"
echo "   - Sign in or create an account"
echo "   - Click 'Create new secret key'"
echo ""

echo "2. Update your .env file:"
echo "   nano .env"
echo "   # Replace the selected provider value with its real secret"
echo ""

echo "3. Run this script again to restart services"
echo ""

# Check that at least one provider key is configured. Format varies by provider.
if grep -Eq '^[[:space:]]*(OPENAI_API_KEY|OPENROUTER_API_KEY|FIREWORKS_API_KEY|NVIDIA_API_KEY)=[^[:space:]]+' "$ENV_FILE"; then
    echo -e "${GREEN}✅ Provider key configuration is present${NC}"
    echo ""
    
    # Ask to restart
    read -p "Restart Docker services to apply changes? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔄 Restarting services..."
        "${COMPOSE_BIN[@]}" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
        echo ""
        echo "🚀 Starting services with new API key..."
        "${COMPOSE_BIN[@]}" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
        echo ""
        echo "⏳ Waiting for services to initialize (40 seconds)..."
        sleep 40
        echo ""
        echo "🏥 Checking health..."
        curl -s http://localhost:8000/health/ | python3 -m json.tool
        echo ""
        echo -e "${GREEN}✅ Services restarted!${NC}"
        echo ""
        echo "📝 Test document upload:"
        echo "   1. Open: http://localhost:8080"
        echo "   2. Upload a PDF"
        echo "   3. Check for success message"
    fi
else
    echo -e "${YELLOW}⚠️  Could not find a configured provider key${NC}"
    echo ""
    echo "Set at least one provider key in $ENV_FILE before restarting."
fi
