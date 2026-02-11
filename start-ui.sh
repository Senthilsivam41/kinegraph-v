#!/bin/bash

# KineGraph Quick Start Script
# Launches both backend API and frontend UI

set -e

echo "🚀 KineGraph Quick Start"
echo "========================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is running
echo -e "${BLUE}Checking backend status...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend API is running${NC}"
else
    echo -e "${YELLOW}⚠️  Backend API not detected at http://localhost:8000${NC}"
    echo -e "${YELLOW}Please start the backend first:${NC}"
    echo "   cd /Users/sendils/work/repo/kinetic-v/kinegraph-v"
    echo "   docker-compose up -d"
    echo "   # OR"
    echo "   uvicorn app.main:app --reload"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}Starting Frontend UI...${NC}"
echo ""

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✅ Python 3 found${NC}"
    echo ""
    echo -e "${GREEN}🌐 Frontend UI Server Starting...${NC}"
    echo -e "${GREEN}📍 URL: http://localhost:8080${NC}"
    echo -e "${GREEN}🔗 Backend: http://localhost:8000${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Start the server
    python3 serve.py
else
    echo -e "${YELLOW}⚠️  Python 3 not found${NC}"
    echo "Please install Python 3 or manually start a web server:"
    echo "   cd frontend"
    echo "   python3 -m http.server 8080"
    exit 1
fi
