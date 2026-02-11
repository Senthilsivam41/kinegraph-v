# KineGraph Frontend - Quick Start Guide

## 🎯 Getting Started in 2 Minutes

### Step 1: Start the Backend

You have two options:

#### Option A: Docker (Recommended)
```bash
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v
docker-compose up -d
```

#### Option B: Local Development
```bash
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v
uvicorn app.main:app --reload
```

Wait for all services to initialize (~30 seconds).

### Step 2: Start the Frontend

#### Quick Launch (From Project Root)
```bash
./start-ui.sh
```

#### Manual Launch
```bash
cd frontend
python3 serve.py
```

### Step 3: Open Your Browser

Navigate to: **http://localhost:8080**

---

## 🎨 What You'll See

### 1. System Status Check
Upon loading, the UI checks:
- ✅ API Server
- ✅ ChromaDB
- ✅ Neo4j
- ✅ Redis

### 2. Sidebar Features
- **Upload Documents**: Drag & drop or select PDFs
- **Query Modes**: 
  - Hybrid (best results)
  - Vector (fast semantic)
  - Graph (relationships)
- **Settings**: Customize API endpoint and result count
- **Live Status**: Real-time service monitoring

### 3. Chat Interface
- Type questions naturally
- See real-time processing
- View results with:
  - Source (vector/graph)
  - Relevance scores
  - Document metadata
  - Content previews

---

## 📝 Try These Examples

### Example 1: Upload a Document
1. Click "Choose File" in the sidebar
2. Select a PDF document
3. Click "Upload PDF"
4. Wait for processing to complete (~30 seconds for a typical document)
5. You'll see a success message in the chat

### Example 2: Ask Questions

**For a research paper:**
```
What are the main findings of this study?
```

**For technical documentation:**
```
How do I configure the authentication system?
```

**For multiple documents:**
```
Compare the approaches discussed in these papers
```

### Example 3: Switch Query Modes

**Try Hybrid Mode** (default):
- Best overall results
- Combines both search strategies

**Try Vector Mode**:
- Fastest response
- Good for general questions
- Based on semantic similarity

**Try Graph Mode**:
- Best for "how are X and Y related?"
- Explores entity connections
- Uses Cypher queries

---

## 🔧 Troubleshooting

### Backend Connection Failed
**Symptom**: Red status indicators, "Backend not running" errors

**Solution**:
```bash
# Check if backend is up
curl http://localhost:8000/health

# If not, start it
cd /Users/sendils/work/repo/kinetic-v/kinegraph-v
docker-compose up -d

# Or for local dev
uvicorn app.main:app --reload
```

### CORS Errors
**Symptom**: Browser console shows CORS errors

**Solution**: The FastAPI backend already has CORS enabled. If you're still seeing errors:
1. Make sure you're not opening `index.html` directly (use a server)
2. Check that API URL in settings is correct

### No Results Returned
**Symptom**: Query completes but shows 0 results

**Solution**:
1. Make sure you've uploaded documents first
2. Wait for processing to complete
3. Try a simpler query
4. Try different query modes

### Upload Fails
**Symptom**: Upload button shows error

**Solution**:
1. Ensure file is a PDF
2. Check backend logs: `docker-compose logs -f app`
3. Verify Redis and Celery workers are running
4. Check disk space

---

## 💡 Tips for Best Results

### Query Tips
1. **Be specific**: "What are the benefits of approach X?" vs "Tell me about it"
2. **Use keywords**: Include domain-specific terms
3. **Try different modes**: Each mode has strengths
4. **Iterate**: Refine based on results

### Upload Tips
1. **PDF quality matters**: Clean, text-based PDFs work best
2. **Multiple docs**: Upload related documents for better context
3. **Metadata**: Add metadata during upload for better filtering
4. **Processing time**: Larger docs take longer (watch the status)

### Performance Tips
1. **Max results**: Lower values (5-10) are faster
2. **Hybrid mode**: Best quality but slightly slower
3. **Vector mode**: Fastest for simple queries
4. **Health checks**: Green = good, investigate red indicators

---

## 🎓 Understanding the Results

### Result Cards Show:
- **Source Badge**: Vector (blue) or Graph (green)
- **Relevance Score**: 0.0 to 1.0 (higher = better match)
- **Content Preview**: First ~200 characters
- **Metadata**: Document name, chunk number, etc.

### Score Interpretation:
- **0.9 - 1.0**: Excellent match
- **0.7 - 0.9**: Good match
- **0.5 - 0.7**: Moderate relevance
- **< 0.5**: Weak match (might not be relevant)

---

## 🚀 Advanced Usage

### Custom API Endpoint
Change the API URL in settings to:
- Local: `http://localhost:8000`
- Docker: `http://localhost:8000`
- Remote: `https://your-domain.com`
- Custom port: `http://localhost:9000`

### Filters (Coming Soon)
Future versions will support:
```json
{
  "query": "machine learning",
  "filters": {
    "author": "John Doe",
    "category": "research",
    "date_range": "2024"
  }
}
```

### Bulk Upload (Manual via API)
Use the API directly for multiple files:
```bash
for file in *.pdf; do
  curl -X POST http://localhost:8000/api/v1/ingest/document \
    -F "file=@$file"
done
```

---

## 📱 Mobile Usage

The UI is fully responsive:
- Works on phones and tablets
- Sidebar collapses on small screens
- Touch-friendly controls
- Optimized scrolling

---

## 🎨 Keyboard Shortcuts

- **Enter**: Send message
- **Shift + Enter**: New line in message
- **Ctrl/Cmd + K**: Focus on input (coming soon)

---

## 🔐 Security Notes

For production:
1. Add authentication to the UI
2. Use HTTPS for all communication
3. Implement rate limiting
4. Restrict CORS to specific origins
5. Add input sanitization
6. Use environment variables for sensitive config

---

## 📊 Performance Benchmarks

Typical performance on modern hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| UI Load | < 1s | Initial page load |
| Health Check | < 100ms | Status indicators |
| Vector Query | 200-500ms | Depends on corpus size |
| Graph Query | 300-800ms | Depends on graph complexity |
| Hybrid Query | 400-1000ms | Combined search |
| PDF Upload | 30-120s | Depends on document size |

---

## 🎯 Next Steps

1. ✅ Upload your first document
2. ✅ Try all three query modes
3. ✅ Explore different types of questions
4. ✅ Check the API documentation
5. ✅ Read the architecture docs
6. ✅ Customize the UI to your needs

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Project README**: [`../README.md`](../README.md)
- **Architecture**: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- **API Guide**: [`../API.md`](../API.md)

---

## 💬 Need Help?

1. Check browser console for errors (F12)
2. Review backend logs: `docker-compose logs -f`
3. Verify all services are running: `docker-compose ps`
4. Test API directly: `curl http://localhost:8000/health`

---

Enjoy using KineGraph! 🚀
