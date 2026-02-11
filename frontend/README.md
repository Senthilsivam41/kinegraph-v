# KineGraph Chat UI

A modern, responsive chat interface for the KineGraph Hybrid RAG System.

## Features

✨ **Intelligent Chat Interface**
- Real-time chat with your documents
- Typing indicators and smooth animations
- Message history with timestamps

🔍 **Three Query Modes**
- **Hybrid Mode** (Recommended): Combines vector and graph search using Reciprocal Rank Fusion
- **Vector Mode**: Fast semantic similarity search using ChromaDB
- **Graph Mode**: Relationship-focused queries using Neo4j

📄 **Document Management**
- Upload PDF documents directly from the UI
- Real-time processing status updates
- Task tracking for document ingestion

⚙️ **Customizable Settings**
- Adjustable result count (1-100)
- Configurable API endpoint
- Query mode selection
- Settings persistence in localStorage

💚 **System Health Monitoring**
- Real-time status of all backend services
- Automatic health checks every 30 seconds
- Visual indicators for service status

## Quick Start

### Prerequisites

Ensure your backend API is running:
```bash
# From the project root
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v
uvicorn app.main:app --reload
```

### Option 1: Using Python's Built-in Server (Recommended)

```bash
# Navigate to the frontend directory
cd frontend

# Start the server
python3 -m http.server 8080
```

Then open your browser to: http://localhost:8080

### Option 2: Using Node.js http-server

```bash
# Install http-server globally (one time)
npm install -g http-server

# Navigate to the frontend directory
cd frontend

# Start the server
http-server -p 8080
```

Then open your browser to: http://localhost:8080

### Option 3: Using VS Code Live Server

1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

### Option 4: Direct File Access

Simply open `index.html` in your browser. Note: some browsers may block API calls due to CORS when opening files directly.

## Configuration

### API URL
By default, the UI connects to `http://localhost:8000`. You can change this in the Settings section of the sidebar.

### Query Settings
- **Max Results**: Control how many results to return (1-100)
- **Mode**: Select between hybrid, vector, or graph mode

## Usage Guide

### 1. Check System Health
Upon loading, the UI automatically checks the health of all backend services:
- API Server
- ChromaDB
- Neo4j
- Redis

### 2. Upload Documents
1. Click "Choose File" and select a PDF
2. Click "Upload PDF"
3. Monitor the processing status
4. Once complete, you can query the document

### 3. Chat with Your Documents
1. Type your question in the chat input
2. Press Enter or click the send button
3. View results with:
   - Source indicator (vector/graph)
   - Relevance score
   - Content preview
   - Document metadata

### 4. Clear Chat History
Click the "🗑️ Clear Chat" button in the header to reset the conversation.

## Query Examples

**Semantic Search (Vector Mode):**
```
What are the main findings in the research?
```

**Relationship Queries (Graph Mode):**
```
How are concepts A and B related?
```

**Comprehensive Search (Hybrid Mode):**
```
Explain the methodology and its connections to previous work
```

## File Structure

```
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Complete styling
├── app.js          # Application logic and API integration
└── README.md       # This file
```

## Architecture

### Components

1. **Chat Interface**
   - Message display with role-based styling
   - Real-time typing indicators
   - Smooth animations and transitions

2. **Sidebar**
   - Document upload
   - Query mode selection
   - Settings configuration
   - System health monitoring

3. **API Integration**
   - RESTful API calls to backend
   - Error handling and retry logic
   - Task polling for async operations

### State Management

The application maintains state for:
- Chat messages and history
- Upload status and file tracking
- User settings (persisted to localStorage)
- System health status

## API Endpoints Used

- `GET /health` - System health check
- `POST /api/v1/query` - Query documents
- `POST /api/v1/ingest/document` - Upload document
- `GET /api/v1/ingest/task/{task_id}` - Check processing status

## Customization

### Styling
Edit `styles.css` to customize:
- Color scheme (CSS variables in `:root`)
- Layout and spacing
- Animations and transitions

### Behavior
Edit `app.js` to customize:
- API endpoints
- Polling intervals
- Result display format
- Error handling

## Troubleshooting

### Backend Connection Issues
1. Verify the backend is running: `curl http://localhost:8000/health`
2. Check the API URL in settings
3. Open browser console for detailed error messages

### CORS Errors
Ensure your backend has CORS properly configured. The FastAPI backend includes CORS middleware by default.

### Upload Failures
1. Verify file is a PDF
2. Check backend logs for processing errors
3. Ensure all services (Redis, ChromaDB, Neo4j) are running

### No Results Returned
1. Upload documents first
2. Try different query modes
3. Check if documents were processed successfully

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern mobile browsers

## Performance

- Lazy loading of results (first 5 shown)
- Efficient DOM updates
- Automatic textarea resizing
- Debounced health checks

## Security Considerations

For production deployment:
1. Change CORS settings in backend to specific origins
2. Use HTTPS for API communication
3. Implement authentication/authorization
4. Add rate limiting
5. Sanitize user inputs

## Future Enhancements

Potential features for future versions:
- [ ] Conversation export/import
- [ ] Multi-file upload
- [ ] Advanced filtering options
- [ ] Dark mode toggle
- [ ] Voice input
- [ ] Markdown rendering in results
- [ ] Document preview
- [ ] Search history

## License

This UI is part of the KineGraph project.

## Support

For issues or questions:
1. Check backend API documentation: `API.md`
2. Review architecture docs: `ARCHITECTURE.md`
3. Check browser console for errors
4. Verify all services are running

---

Built with ❤️ using vanilla JavaScript for maximum performance and minimal dependencies.
