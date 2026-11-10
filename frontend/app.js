// Application State
const state = {
    apiUrl: window.location.protocol === 'https:' ? window.location.origin : 'http://localhost:8000',
    mode: 'hybrid',
    maxResults: 10,
    messages: [],
    uploadedFiles: [],
    attachment: null
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
    modeDescription: document.getElementById('modeDescription'),
    attachmentArea: document.getElementById('attachmentArea'),
    attachmentName: document.getElementById('attachmentName'),
    attachmentSize: document.getElementById('attachmentSize'),
    removeAttachmentBtn: document.getElementById('removeAttachmentBtn'),
    attachBtn: document.getElementById('attachBtn'),
    attachmentInput: document.getElementById('attachmentInput')
};

// Mode Descriptions
const modeDescriptions = {
    hybrid: '<strong>Hybrid Mode:</strong> Combines vector and graph search for best results using Reciprocal Rank Fusion (RRF)',
    vector: '<strong>Vector Mode:</strong> Semantic similarity search using ChromaDB embeddings - fast and good for general queries',
    graph: '<strong>Graph Mode:</strong> Cypher query generation using Neo4j - best for relationship and connection queries',
    vectorless: '<strong>Vectorless Mode:</strong> Pure lexical search (BM25) over documents and attachments without expensive vector databases or embeddings'
};

// Chat Header Titles
const chatHeaderTitles = {
    hybrid: 'Hybrid RAG Chat',
    vector: 'Vector Search Chat',
    graph: 'Graph Search Chat',
    vectorless: 'Vectorless RAG Chat'
};

// Initialize Application
async function init() {
    setupEventListeners();
    await checkSystemHealth();
    loadSettings();
    const active = localStorage.getItem('kinegraph-active-ingest');
    if (active) {
        try {
            const { docId, fileName } = JSON.parse(active);
            if (docId && fileName) pollDocumentStatus(docId, fileName);
        } catch (_) {
            localStorage.removeItem('kinegraph-active-ingest');
        }
    }
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
            
            // Update chat header text dynamically
            const chatHeader = document.querySelector('.chat-header h2');
            if (chatHeader) {
                chatHeader.textContent = chatHeaderTitles[state.mode];
            }
            
            saveSettings();
        });
    });
    
    // Auto-resize textarea
    elements.chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // Attachment triggers
    elements.attachBtn.addEventListener('click', () => elements.attachmentInput.click());
    elements.attachmentInput.addEventListener('change', handleAttachmentSelection);
    elements.removeAttachmentBtn.addEventListener('click', removeAttachment);
}

// Handle Input Change
function handleInputChange() {
    const hasText = elements.chatInput.value.trim().length > 0;
    const hasAttachment = state.attachment !== null;
    elements.sendBtn.disabled = !hasText && !hasAttachment;
}

// Handle Attachment Selection
function handleAttachmentSelection(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(evt) {
        state.attachment = {
            name: file.name,
            size: file.size,
            content: evt.target.result
        };
        
        // Show badge
        elements.attachmentName.textContent = file.name;
        const kbSize = (file.size / 1024).toFixed(1);
        elements.attachmentSize.textContent = `(${kbSize} KB)`;
        elements.attachmentArea.style.display = 'block';
        elements.sendBtn.disabled = false; // Enable send button when attachment is added
        
        console.log(`Attached file: ${file.name} (${file.size} bytes)`);
    };
    reader.onerror = function() {
        alert('Failed to read the file.');
    };
    reader.readAsText(file);
}

// Remove Attachment
function removeAttachment() {
    state.attachment = null;
    elements.attachmentInput.value = '';
    elements.attachmentArea.style.display = 'none';
    handleInputChange();
}

// Send Message
async function handleSendMessage() {
    const query = elements.chatInput.value.trim();
    if (!query && !state.attachment) return;
    
    // Add user message
    const userMsgContent = query + (state.attachment ? `\n\n📎 Attached: ${state.attachment.name}` : '');
    addMessage('user', userMsgContent);
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    elements.sendBtn.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();

    const payload = {
        query: query || `Summarize the attached file: ${state.attachment.name}`,
        mode: state.mode,
        max_results: state.maxResults
    };

    if (state.attachment) {
        payload.attachment_content = state.attachment.content;
        payload.attachment_name = state.attachment.name;
        removeAttachment();
    }
    
    try {
        // Call query API
        const response = await fetch(`${state.apiUrl}/api/v1/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add assistant message with results
        addAssistantMessage(data, payload.mode);
        
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

const retrievalLabels = new Map([
    ['hybrid', 'Hybrid'],
    ['vector', 'Vector'],
    ['graph', 'Graph'],
    ['lexical', 'Lexical'],
    ['vectorless', 'Vectorless']
]);

function retrievalLabel(mode) {
    return retrievalLabels.get(mode) ?? mode;
}

function resultElement(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
}

function resultDetails(label, className) {
    const details = resultElement('details', null, className);
    details.append(resultElement('summary', label));
    return details;
}

function retrievalSummary(data, requestMode, resultCount) {
    const summary = resultElement('div', null, 'query-retrieval');
    const requested = retrievalLabel(data.requested_mode ?? data.mode ?? requestMode);
    const effective = retrievalLabel(data.effective_mode);
    const count = data.total_results ?? resultCount;
    const timing = Number.isFinite(data.execution_time_ms) ? ` in ${data.execution_time_ms} ms` : '';
    const mode = effective ? ` using ${effective} search` : '';
    const requestedMode = requested && requested !== effective ? ` (requested ${requested})` : '';
    summary.append(resultElement('p', `Found ${count} result${count === 1 ? '' : 's'}${mode}${requestedMode}${timing}.`));
    if (!effective) summary.append(resultElement('p', 'Effective search mode unavailable.'));

    const routing = data.routing_details ?? {};
    const recovery = data.recovery_details ?? {};
    const escalation = recovery.route_escalation ?? routing.route_escalation;
    if (escalation?.triggered) {
        const from = retrievalLabel(escalation.from_mode);
        const to = retrievalLabel(escalation.to_mode);
        const route = from && to ? ` from ${from} to ${to}` : '';
        const reasons = (escalation.trigger_reasons ?? []).map(reason => reason.replaceAll('_', ' '));
        const reason = reasons.join(', ') || routing.decision;
        summary.append(resultElement('p', `Retrieval switched${route} during recovery.${reason ? ` Reason: ${reason}` : ''}`));
    } else if (requested && effective && requested !== effective) {
        const reason = routing.decision ?? routing.execution_plan?.decision;
        summary.append(resultElement('p', reason ? `Routing reason: ${reason}` : 'Routing reason unavailable.'));
    }

    if (recovery.structured_recovery_used || recovery.hyde_used || (data.recovery_triggered && !escalation?.triggered)) {
        const reasons = (recovery.initial_assessment?.reasons ?? []).map(reason => reason.replaceAll('_', ' '));
        summary.append(resultElement('p', `Additional retrieval was attempted${effective ? ` in ${effective} mode` : ''}.${reasons.length ? ` Reason: ${reasons.join(', ')}` : ''}`));
    }
    return summary;
}

function createQueryAnswer(data, claims, citationTargets, hasResults) {
    const answer = resultElement('div', null, 'generated-answer');
    if (!claims.length) answer.classList.add('answer-without-claims');
    answer.append(resultElement('strong', 'Answer'));
    if (claims.length) {
        for (const claim of claims) {
            const paragraph = resultElement('p', claim.text, 'answer-claim');
            for (const id of new Set(claim.chunk_ids ?? [])) {
                const target = citationTargets.get(id);
                if (target) {
                    const citation = resultElement('button', `[${target.number}]`, 'query-citation');
                    citation.type = 'button';
                    citation.setAttribute('aria-label', `Open cited passage ${target.number} from ${target.source}`);
                    citation.addEventListener('click', target.open);
                    paragraph.append(citation);
                } else {
                    paragraph.append(resultElement('span', `[${id}: passage unavailable]`, 'query-citation-unavailable'));
                }
            }
            answer.append(paragraph);
        }
    } else {
        if (data.generated_answer) {
            answer.append(resultElement('p', data.generated_answer, 'answer-claim'));
        }
        if (hasResults) {
            const status = Array.isArray(data.grounded_claims)
                ? 'No grounded claims were returned. Retrieved passages are still available below.'
                : 'Claim citations are unavailable for this response. Retrieved passages are still available below.';
            answer.append(resultElement('p', status, 'query-evidence-note'));
        }
    }

    const validation = data.citation_validation;
    if (validation?.structured_output_valid && validation.total_claims > 0) {
        const rejected = validation.rejected_claims?.length ?? 0;
        const status = rejected
            ? `Citation check rejected ${rejected} claim${rejected === 1 ? '' : 's'}.`
            : 'Citation check found no invalid or missing references in the generated claims.';
        answer.append(resultElement('p', status, 'query-evidence-note'));
    } else if (validation) {
        answer.append(resultElement('p', 'Citation check unavailable for this answer.', 'query-evidence-note'));
    }

    const review = data.grounding_critique;
    const coverage = data.answer_relevancy?.question_coverage ?? review?.question_coverage;
    if (review?.completed) {
        const label = ['complete', 'partial', 'none'].includes(coverage) ? coverage : 'unverified';
        answer.append(resultElement('p', `Grounding review completed. Question coverage: ${label}.`, 'query-evidence-note'));
    } else if (claims.length || review) {
        answer.append(resultElement('p', 'Grounding review unavailable for this answer.', 'query-evidence-note'));
    }
    if (coverage === 'partial' || coverage === 'none') {
        const facets = data.answer_relevancy?.missing_question_facets ?? review?.missing_question_facets ?? [];
        if (!review?.completed) {
            answer.append(resultElement('p', `Reported question coverage: ${coverage}.`, 'query-evidence-note'));
        }
        if (facets.length) {
            answer.append(resultElement('p', `Missing coverage: ${facets.join(' · ')}`, 'query-evidence-note'));
        }
    }
    if (Number.isFinite(data.answer_confidence)) {
        const level = data.answer_confidence >= 0.7 ? 'high' : 'low';
        answer.append(resultElement('div', `Confidence: ${(data.answer_confidence * 100).toFixed(0)}%`, `confidence-badge ${level}`));
    }
    return answer;
}

function createRankingExplanation(explanation) {
    const details = resultDetails('Why this result?', 'result-explanation');
    const channels = explanation?.source_channels ?? [];
    if (!channels.length) {
        details.append(resultElement('p', 'Retrieval explanation unavailable for this passage.'));
        return details;
    }

    details.append(resultElement('p', `Found by ${channels.map(retrievalLabel).join(' and ')} search.`));
    const ranks = resultElement('ul');
    for (const channel of channels) {
        const rank = explanation.channel_ranks?.[channel];
        const score = explanation.original_scores?.[channel];
        const contribution = explanation.rrf_contributions?.[channel];
        const fields = [
            Number.isFinite(rank) ? `rank ${rank} before merging` : null,
            Number.isFinite(score) ? `retrieval score ${score}` : null,
            Number.isFinite(contribution) ? `contributed ${contribution} to the merged score` : null
        ].filter(Boolean);
        if (fields.length) ranks.append(resultElement('li', `${retrievalLabel(channel)}: ${fields.join(', ')}`));
    }
    if (ranks.childElementCount) details.append(ranks);

    const reranking = explanation.reranking;
    if (reranking?.mode && reranking.mode !== 'unknown') {
        const components = Object.entries(reranking.components ?? {})
            .filter(([, value]) => Number.isFinite(value))
            .map(([key, value]) => `${key.replaceAll('_', ' ')} ${value}`);
        const score = Number.isFinite(reranking.score) ? `, score ${reranking.score}` : '';
        const breakdown = components.length ? ` (${components.join(', ')})` : '';
        details.append(resultElement('p', `Reranking: ${reranking.mode.replaceAll('_', ' ')}${score}${breakdown}.`));
    }

    for (const path of Object.values(explanation.graph_paths ?? {})) {
        if (!Array.isArray(path)) continue;
        const steps = path.map(step => {
            const relation = step?.relationship_type ?? step?.relationship;
            if (!relation) return null;
            if (!step.from_node_id || !step.to_node_id) return relation;
            const left = step.direction === 'INCOMING' ? '←' : '—';
            const right = step.direction === 'OUTGOING' ? '→' : '—';
            return `${step.from_node_id} ${left} ${relation} ${right} ${step.to_node_id}`;
        }).filter(Boolean);
        if (steps.length) details.append(resultElement('p', `Graph path: ${steps.join(' · ')}`));
    }
    return details;
}

function createResultCard(result, number) {
    const card = resultElement('div', null, 'result-card');
    const metadata = result.metadata ?? {};
    const sourceName = metadata.file_name || `result ${number}`;
    const header = resultElement('div', null, 'result-header');
    const source = resultElement('span', `Result ${number} · ${retrievalLabel(result.source) ?? 'Unknown source'}`, 'result-source');
    if (retrievalLabels.has(result.source)) source.classList.add(result.source);
    header.append(source);
    if (Number.isFinite(result.score)) {
        header.append(resultElement('span', `Score: ${result.score.toFixed(3)}`, 'result-score'));
    }
    card.append(header);

    const location = [];
    if (metadata.file_name) location.push(metadata.file_name);
    if (metadata.section_path) location.push(metadata.section_path);
    if (metadata.page_start > 0) {
        const end = metadata.page_end;
        location.push(end > metadata.page_start ? `Pages ${metadata.page_start}–${end}` : `Page ${metadata.page_start}`);
    }
    if (Number.isInteger(metadata.chunk_index) && metadata.chunk_index >= 0) {
        const total = metadata.total_chunks > 0 ? `/${metadata.total_chunks}` : '';
        location.push(`Chunk ${metadata.chunk_index + 1}${total}`);
    }
    if (location.length) card.append(resultElement('p', location.join(' · '), 'result-metadata'));
    const passage = resultDetails('Read passage', 'result-passage');
    const text = result.content || 'Passage text unavailable.';
    const preview = text.length > 200 ? `${text.slice(0, 200)}…` : text;
    passage.firstElementChild.append(resultElement('span', preview, 'result-preview'));
    const fullText = resultElement('p', text, 'result-passage-text');
    fullText.tabIndex = -1;
    passage.append(fullText);
    card.append(passage, createRankingExplanation(result.explanation));
    return { card, sourceName, passage, fullText };
}

function addAssistantMessage(data, requestMode) {
    const messageDiv = resultElement('div', null, 'message assistant');
    const contentDiv = resultElement('div', null, 'message-content');
    const results = data.results ?? [];
    const claims = (data.grounded_claims ?? []).filter(claim => claim?.text);
    const citedIds = new Set(claims.flatMap(claim => claim.chunk_ids ?? []));
    const citationTargets = new Map();
    const visibleCards = [];
    const initialResultLimit = 5;
    const extraResultCount = results.length - initialResultLimit;
    const moreResults = extraResultCount > 0
        ? resultDetails(`Show ${extraResultCount} more result${extraResultCount === 1 ? '' : 's'}`, 'query-more-results')
        : null;

    results.forEach((result, index) => {
        const number = index + 1;
        const id = result.metadata?.citation_id;
        const inMoreResults = index >= initialResultLimit;
        const { card, sourceName, passage, fullText } = createResultCard(result, number);
        if (inMoreResults) moreResults.append(card);
        else visibleCards.push(card);
        if (id && citedIds.has(id)) {
            // An ambiguous reference should never open the wrong passage.
            citationTargets.set(id, citationTargets.has(id) ? null : {
                number,
                source: sourceName,
                passage,
                open() {
                    if (inMoreResults) moreResults.open = true;
                    passage.open = true;
                    fullText.focus();
                }
            });
        }
    });
    for (const target of citationTargets.values()) {
        target?.passage.before(resultElement('p', 'Cited in the answer', 'result-cited'));
    }

    contentDiv.append(retrievalSummary(data, requestMode, results.length));
    if (claims.length || data.generated_answer || results.length) {
        contentDiv.append(createQueryAnswer(data, claims, citationTargets, results.length > 0));
    }
    contentDiv.append(...visibleCards);
    if (moreResults) contentDiv.append(moreResults);
    if (!results.length) {
        contentDiv.append(resultElement('p', 'No passages were retrieved. Try adjusting your query or uploading a document.', 'query-evidence-note'));
    }

    messageDiv.append(contentDiv, resultElement('div', new Date().toLocaleTimeString(), 'message-meta'));
    elements.chatMessages.querySelector('.welcome-message')?.remove();
    elements.chatMessages.append(messageDiv);
    scrollToBottom();

    state.messages.push({ role: 'assistant', content: data, timestamp: Date.now() });
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
    
    if (!/\.(pdf|docx|txt)$/i.test(file.name)) {
        showUploadStatus('Only PDF, DOCX and TXT files are supported', 'error');
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
        
        if (data.doc_id) {
            localStorage.setItem('kinegraph-active-ingest', JSON.stringify({docId: data.doc_id, fileName: file.name}));
            pollDocumentStatus(data.doc_id, file.name);
        } else {
            await pollTaskStatus(data.task_id, file.name);
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus(`Error: ${error.message}`, 'error');
        elements.uploadBtn.disabled = false;
    }
}

async function pollDocumentStatus(docId, fileName) {
    const url = `${state.apiUrl}/api/v1/ingest/documents/${encodeURIComponent(docId)}`;
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Status unavailable (${response.status})`);
        const data = await response.json();
        if (data.state === 'done' && data.validation?.complete) {
            localStorage.removeItem('kinegraph-active-ingest');
            showUploadStatus(`✅ ${fileName} indexed and ready`, 'success');
            state.uploadedFiles.push(fileName);
            elements.fileInput.value = '';
            elements.uploadBtn.disabled = false;
            addMessage('assistant', `Document "${fileName}" has been successfully indexed. You can now ask questions about it!`);
            return;
        }
        if (data.state === 'failed' || data.state === 'done') {
            localStorage.removeItem('kinegraph-active-ingest');
            showUploadStatus(`❌ ${fileName}: ${data.error_code || 'index validation incomplete'}`, 'error');
            elements.uploadBtn.disabled = false;
            return;
        }
        showUploadStatus(`Processing ${fileName}… ${data.stage}`, 'processing');
    } catch (error) {
        showUploadStatus(`Waiting for status service: ${error.message}`, 'processing');
    }
    setTimeout(() => pollDocumentStatus(docId, fileName), 5000);
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
                <p><strong>Get started:</strong> Upload a PDF, DOCX, or TXT document or ask me a question!</p>
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
            
            // Update chat header text dynamically
            const chatHeader = document.querySelector('.chat-header h2');
            if (chatHeader) {
                chatHeader.textContent = chatHeaderTitles[state.mode];
            }
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
