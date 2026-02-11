// Application State
const state = {
    apiUrl: 'http://localhost:8000',
    mode: 'hybrid',
    maxResults: 10,
    messages: [],
    uploadedFiles: []
};

// DOM Elements
const elements = {
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    clearChat: document.getElementById('clearChat'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    uploadStatus: document.getElementById('uploadStatus'),
    systemStatus: document.getElementById('systemStatus'),
    apiUrl: document.getElementById('apiUrl'),
    maxResults: document.getElementById('maxResults'),
    modeDescription: document.getElementById('modeDescription')
};

// Mode Descriptions
const modeDescriptions = {
    hybrid: '<strong>Hybrid Mode:</strong> Combines vector and graph search for best results using Reciprocal Rank Fusion (RRF)',
    vector: '<strong>Vector Mode:</strong> Semantic similarity search using ChromaDB embeddings - fast and good for general queries',
    graph: '<strong>Graph Mode:</strong> Cypher query generation using Neo4j - best for relationship and connection queries'
};

// Initialize Application
async function init() {
    setupEventListeners();
    await checkSystemHealth();
    loadSettings();
    console.log('KineGraph Chat UI initialized');
}

// Event Listeners
function setupEventListeners() {
    // Chat input handlers
    elements.sendBtn.addEventListener('click', handleSendMessage);
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    elements.chatInput.addEventListener('input', handleInputChange);
    
    // Clear chat
    elements.clearChat.addEventListener('click', clearChat);
    
    // File upload
    elements.uploadBtn.addEventListener('click', handleFileUpload);
    elements.fileInput.addEventListener('change', () => {
        elements.uploadBtn.disabled = !elements.fileInput.files.length;
    });
    
    // Settings
    elements.apiUrl.addEventListener('change', (e) => {
        state.apiUrl = e.target.value;
        saveSettings();
        checkSystemHealth();
    });
    
    elements.maxResults.addEventListener('change', (e) => {
        state.maxResults = parseInt(e.target.value);
        saveSettings();
    });
    
    // Mode selection
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.mode = e.target.value;
            elements.modeDescription.innerHTML = modeDescriptions[state.mode];
            saveSettings();
        });
    });
    
    // Auto-resize textarea
    elements.chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

// Handle Input Change
function handleInputChange() {
    const hasText = elements.chatInput.value.trim().length > 0;
    elements.sendBtn.disabled = !hasText;
}

// Send Message
async function handleSendMessage() {
    const query = elements.chatInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage('user', query);
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    elements.sendBtn.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Call query API
        const response = await fetch(`${state.apiUrl}/api/v1/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                mode: state.mode,
                max_results: state.maxResults
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add assistant message with results
        addAssistantMessage(data);
        
    } catch (error) {
        console.error('Query error:', error);
        removeTypingIndicator(typingId);
        addMessage('assistant', `❌ Error: ${error.message}. Please check if the backend is running.`);
    }
}

// Add Message
function addMessage(role, content, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    
    // Remove welcome message if exists
    const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Store in state
    state.messages.push({ role, content, timestamp: Date.now(), metadata });
}

// Add Assistant Message with Results
function addAssistantMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Summary
    const summary = document.createElement('p');
    summary.innerHTML = `Found <strong>${data.total_results}</strong> result${data.total_results !== 1 ? 's' : ''} using <strong>${data.mode}</strong> mode in <strong>${data.execution_time_ms}ms</strong>`;
    contentDiv.appendChild(summary);
    
    // Results
    if (data.results && data.results.length > 0) {
        data.results.slice(0, 5).forEach((result, index) => {
            const resultCard = createResultCard(result, index);
            contentDiv.appendChild(resultCard);
        });
        
        if (data.results.length > 5) {
            const moreInfo = document.createElement('p');
            moreInfo.style.marginTop = '12px';
            moreInfo.style.fontSize = '13px';
            moreInfo.style.fontStyle = 'italic';
            moreInfo.textContent = `... and ${data.results.length - 5} more results`;
            contentDiv.appendChild(moreInfo);
        }
    } else {
        const noResults = document.createElement('p');
        noResults.style.marginTop = '12px';
        noResults.textContent = 'No results found. Try uploading documents or adjusting your query.';
        contentDiv.appendChild(noResults);
    }
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    
    // Remove welcome message if exists
    const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Store in state
    state.messages.push({ 
        role: 'assistant', 
        content: data, 
        timestamp: Date.now() 
    });
}

// Create Result Card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const header = document.createElement('div');
    header.className = 'result-header';
    
    const source = document.createElement('span');
    source.className = `result-source ${result.source}`;
    source.textContent = result.source;
    
    const score = document.createElement('span');
    score.className = 'result-score';
    score.textContent = `Score: ${result.score.toFixed(3)}`;
    
    header.appendChild(source);
    header.appendChild(score);
    
    const content = document.createElement('div');
    content.className = 'result-content';
    content.textContent = result.content.substring(0, 200) + (result.content.length > 200 ? '...' : '');
    
    const metadata = document.createElement('div');
    metadata.className = 'result-metadata';
    const metaText = [];
    if (result.metadata.file_name) metaText.push(`📄 ${result.metadata.file_name}`);
    if (result.metadata.chunk_index !== undefined) metaText.push(`Chunk ${result.metadata.chunk_index + 1}/${result.metadata.total_chunks}`);
    metadata.textContent = metaText.join(' • ');
    
    card.appendChild(header);
    card.appendChild(content);
    if (metaText.length > 0) card.appendChild(metadata);
    
    return card;
}

// Typing Indicator
function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    const id = 'typing-' + Date.now();
    messageDiv.id = id;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingDiv.appendChild(dot);
    }
    
    messageDiv.appendChild(typingDiv);
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// File Upload
async function handleFileUpload() {
    const file = elements.fileInput.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.pdf')) {
        showUploadStatus('Only PDF files are supported', 'error');
        return;
    }
    
    elements.uploadBtn.disabled = true;
    showUploadStatus('Uploading...', 'processing');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify({
            uploaded_at: new Date().toISOString(),
            source: 'chat-ui'
        }));
        
        const response = await fetch(`${state.apiUrl}/api/v1/ingest/document`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Poll task status
        await pollTaskStatus(data.task_id, file.name);
        
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus(`Error: ${error.message}`, 'error');
        elements.uploadBtn.disabled = false;
    }
}

// Poll Task Status
async function pollTaskStatus(taskId, fileName) {
    const maxAttempts = 60; // 5 minutes max
    let attempts = 0;
    
    const poll = async () => {
        try {
            const response = await fetch(`${state.apiUrl}/api/v1/ingest/task/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'SUCCESS') {
                showUploadStatus(`✅ ${fileName} processed successfully!`, 'success');
                state.uploadedFiles.push(fileName);
                elements.fileInput.value = '';
                elements.uploadBtn.disabled = false;
                
                // Add system message to chat
                addMessage('assistant', `Document "${fileName}" has been successfully processed and indexed. You can now ask questions about it!`);
                
            } else if (data.status === 'FAILURE') {
                showUploadStatus(`❌ Processing failed: ${data.error || 'Unknown error'}`, 'error');
                elements.uploadBtn.disabled = false;
                
            } else if (attempts < maxAttempts) {
                // Still processing
                showUploadStatus(`Processing ${fileName}... (${data.status})`, 'processing');
                attempts++;
                setTimeout(poll, 5000); // Poll every 5 seconds
                
            } else {
                showUploadStatus('Processing timeout. Check backend logs.', 'error');
                elements.uploadBtn.disabled = false;
            }
            
        } catch (error) {
            console.error('Poll error:', error);
            showUploadStatus(`Error checking status: ${error.message}`, 'error');
            elements.uploadBtn.disabled = false;
        }
    };
    
    poll();
}

// Show Upload Status
function showUploadStatus(message, type) {
    elements.uploadStatus.textContent = message;
    elements.uploadStatus.className = `upload-status ${type}`;
    elements.uploadStatus.style.display = 'block';
}

// Check System Health
async function checkSystemHealth() {
    try {
        const response = await fetch(`${state.apiUrl}/health`);
        const data = await response.json();
        
        displaySystemStatus(data);
        
    } catch (error) {
        console.error('Health check error:', error);
        displaySystemStatus({
            status: 'error',
            services: {
                api: false,
                chroma: false,
                neo4j: false,
                redis: false
            }
        });
    }
}

// Display System Status
function displaySystemStatus(health) {
    elements.systemStatus.innerHTML = '';
    
    const services = health.services || {};
    const serviceNames = {
        api: 'API Server',
        chroma: 'ChromaDB',
        neo4j: 'Neo4j',
        redis: 'Redis'
    };
    
    Object.entries(serviceNames).forEach(([key, name]) => {
        const item = document.createElement('div');
        item.className = 'status-item';
        
        const dot = document.createElement('span');
        dot.className = `status-dot ${services[key] ? 'online' : 'offline'}`;
        
        const text = document.createElement('span');
        text.textContent = `${name}: ${services[key] ? 'Online' : 'Offline'}`;
        
        item.appendChild(dot);
        item.appendChild(text);
        elements.systemStatus.appendChild(item);
    });
}

// Clear Chat
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        state.messages = [];
        elements.chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>👋 Welcome to KineGraph!</h2>
                <p>I'm your hybrid RAG assistant. I can help you:</p>
                <ul>
                    <li>Search through your documents using semantic similarity</li>
                    <li>Explore relationships and connections in your knowledge graph</li>
                    <li>Answer questions using both vector and graph databases</li>
                </ul>
                <p><strong>Get started:</strong> Upload a PDF document or ask me a question!</p>
            </div>
        `;
    }
}

// Scroll to Bottom
function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Settings Management
function saveSettings() {
    localStorage.setItem('kinegraph-settings', JSON.stringify({
        apiUrl: state.apiUrl,
        mode: state.mode,
        maxResults: state.maxResults
    }));
}

function loadSettings() {
    const saved = localStorage.getItem('kinegraph-settings');
    if (saved) {
        try {
            const settings = JSON.parse(saved);
            state.apiUrl = settings.apiUrl || state.apiUrl;
            state.mode = settings.mode || state.mode;
            state.maxResults = settings.maxResults || state.maxResults;
            
            // Update UI
            elements.apiUrl.value = state.apiUrl;
            elements.maxResults.value = state.maxResults;
            document.querySelector(`input[name="mode"][value="${state.mode}"]`).checked = true;
            elements.modeDescription.innerHTML = modeDescriptions[state.mode];
        } catch (e) {
            console.error('Error loading settings:', e);
        }
    }
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Periodic health check
setInterval(checkSystemHealth, 30000); // Every 30 seconds
