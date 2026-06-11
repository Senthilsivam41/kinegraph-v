#!/bin/bash

# Script to update OpenAI API key and restart services

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"
if docker compose version >/dev/null 2>&1; then
    COMPOSE_BIN=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_BIN=(docker-compose)
else
    echo "Docker Compose v2 or docker-compose is required to restart services."
    exit 1
fi

echo "🔧 KineGraph - OpenAI API Key Setup"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}⚠️  Current Issue:${NC}"
echo "You have a Google API key (AIzaSy...) instead of an OpenAI API key"
echo ""

echo -e "${YELLOW}📋 Steps to Fix:${NC}"
echo ""
echo "1. Get an OpenAI API key from: https://platform.openai.com/api-keys"
echo "   - Sign in or create an account"
echo "   - Click 'Create new secret key'"
echo "   - Copy the key (starts with 'sk-')"
echo ""

echo "2. Update your .env file:"
echo "   nano .env"
echo "   # Find the line: OPENAI_API_KEY=AIzaSyBO..."
echo "   # Replace with: OPENAI_API_KEY=sk-your-actual-key"
echo ""

echo "3. Run this script again to restart services"
echo ""

# Check if OpenAI key is still wrong
if grep -q "OPENAI_API_KEY=AIzaSy" .env; then
    echo -e "${RED}❌ Still using Google API key${NC}"
    echo ""
    echo "Please update your .env file with a valid OpenAI API key"
    echo "OpenAI keys start with: sk-"
    echo ""
    echo "Get your key here: https://platform.openai.com/api-keys"
    exit 1
fi

# Check if key looks valid
if grep -q "OPENAI_API_KEY=sk-" .env; then
    echo -e "${GREEN}✅ OpenAI API key format looks correct${NC}"
    echo ""
    
    # Ask to restart
    read -p "Restart Docker services to apply changes? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔄 Restarting services..."
        "${COMPOSE_BIN[@]}" -f "$COMPOSE_FILE" down
        echo ""
        echo "🚀 Starting services with new API key..."
        "${COMPOSE_BIN[@]}" -f "$COMPOSE_FILE" up -d
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
    echo -e "${YELLOW}⚠️  Could not find a valid OpenAI API key${NC}"
    echo ""
    echo "Make sure your .env file has:"
    echo "OPENAI_API_KEY=sk-your-actual-key-here"
fi
