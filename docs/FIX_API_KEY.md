# Fix: OpenAI API Key Issue

## 🔍 Problem Identified

Your document uploads are failing because you're using a **Google API key** instead of an **OpenAI API key**.

### Error in Logs:
```
Error code: 401 - Incorrect API key provided: AIzaSyBO***fVvw
```

### Current Configuration:
```bash
OPENAI_API_KEY=AIzaSyBO5kPUxn3p74SHS3a8cm4_8hwPS8XfVvw  # ❌ Google API key
```

### Why This Matters:
The system uses OpenAI's API to generate embeddings (vector representations) of your documents. Without a valid OpenAI API key, documents cannot be processed and stored in ChromaDB.

---

## ✅ Solution

### Option 1: Quick Fix Script (Recommended)

1. **Get your OpenAI API key**:
   - Visit: https://platform.openai.com/api-keys
   - Sign in or create an account
   - Click "Create new secret key"
   - Copy the key (starts with `sk-`)

2. **Update `.env` file**:
   ```bash
   nano .env
   ```
   
   Find and replace:
   ```bash
   # From:
   OPENAI_API_KEY=AIzaSyBO5kPUxn3p74SHS3a8cm4_8hwPS8XfVvw
   
   # To:
   OPENAI_API_KEY=sk-proj-your-actual-openai-key-here
   ```

3. **Run the fix script**:
   ```bash
   ./fix-api-key.sh
   ```
   
   This will:
   - Verify your key format
   - Restart Docker services
   - Check system health
   - Confirm everything is working

### Option 2: Manual Fix

```bash
# 1. Update .env file
nano .env
# Replace the Google key with your OpenAI key

# 2. Restart services
docker-compose down
docker-compose up -d

# 3. Wait for services to start (40 seconds)
sleep 40

# 4. Verify health
curl http://localhost:8000/health/
```

---

## 🔑 How to Get an OpenAI API Key

### Step-by-Step:

1. **Go to OpenAI Platform**
   - URL: https://platform.openai.com/api-keys
   - If you don't have an account, sign up (free)

2. **Create API Key**
   - Click "+ Create new secret key"
   - Give it a name (e.g., "KineGraph")
   - Select permissions (default is fine)
   - Click "Create secret key"

3. **Copy the Key**
   - **IMPORTANT**: Copy it immediately
   - You won't be able to see it again
   - Save it somewhere safe
   - Format: `sk-proj-...` or `sk-...`

4. **Verify You Have Credit**
   - Check: https://platform.openai.com/usage
   - New accounts get free credits
   - Or add a payment method

---

## 💰 Pricing Information

OpenAI's `text-embedding-ada-002` model (used by this system):
- **Cost**: $0.0004 per 1,000 tokens (~750 words)
- **Example**: Processing a 100-page PDF ≈ $0.10-$0.30

**Free credits**: New accounts typically get $5-$18 in free credits.

---

## ✨ After Fixing

Once you've updated your API key and restarted services, you can:

1. **Test with Chat UI**:
   - Open: http://localhost:8080
   - Upload a PDF document
   - Wait for "✅ Document processed successfully!"
   - Ask questions about the document

2. **Verify in Logs**:
   ```bash
   # Should see successful processing
   docker-compose logs worker -f
   ```

3. **Check ChromaDB**:
   ```bash
   # Should return documents after upload
   curl -X POST http://localhost:8000/api/v1/query/ \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "mode": "vector"}'
   ```

---

## 🐛 Troubleshooting

### "Invalid API Key" Error Persists
```bash
# Check what's in your .env
grep OPENAI_API_KEY .env

# Make sure it starts with 'sk-'
# Not 'AIzaSy' (Google) or other formats
```

### Services Won't Restart
```bash
# Force clean restart
docker-compose down -v
docker-compose up -d --build
```

### Key Works But Still Failing
```bash
# Check OpenAI account status
# Visit: https://platform.openai.com/usage
# Ensure you have available credit

# Check rate limits
# New accounts have usage limits
```

---

## 📝 Common Mistakes

❌ **Using the wrong type of key**:
- Google API keys: `AIzaSy...`
- Azure OpenAI: Different format
- OpenAI org key: Different from API key

✅ **Correct OpenAI API key format**:
- `sk-proj-...` (new project keys)
- `sk-...` (older format)

❌ **Spaces or quotes in .env**:
```bash
OPENAI_API_KEY="sk-..." # ❌ Don't use quotes
OPENAI_API_KEY= sk-...  # ❌ No space after =
```

✅ **Correct format**:
```bash
OPENAI_API_KEY=sk-proj-your-actual-key
```

---

## 🔐 Security Best Practices

1. **Never commit `.env` to Git**:
   ```bash
   # Already in .gitignore, but verify:
   git status
   # Should NOT show .env file
   ```

2. **Rotate keys regularly**:
   - Create new keys periodically
   - Delete old keys from OpenAI dashboard

3. **Use different keys for dev/prod**:
   - Development: One key
   - Production: Separate key with monitoring

4. **Monitor usage**:
   - Check: https://platform.openai.com/usage
   - Set up usage alerts

---

## ✅ Success Checklist

After fixing the API key:

- [ ] `.env` file has valid OpenAI key (starts with `sk-`)
- [ ] Services restarted: `docker-compose ps` shows all healthy
- [ ] Health check passes: `curl http://localhost:8000/health/`
- [ ] Upload works: Try uploading a PDF via UI
- [ ] Processing succeeds: Check worker logs for success
- [ ] Queries work: Ask questions about uploaded documents

---

## 🎯 Quick Commands Reference

```bash
# Check current key
grep OPENAI_API_KEY .env

# Edit key
nano .env

# Restart services
docker-compose down && docker-compose up -d

# Check health
curl http://localhost:8000/health/

# Watch worker logs
docker-compose logs worker -f

# Test upload via UI
open http://localhost:8080
```

---

## 📞 Still Having Issues?

If problems persist after updating the key:

1. **Verify key is active**: https://platform.openai.com/api-keys
2. **Check account has credit**: https://platform.openai.com/usage
3. **Try a new key**: Create fresh key to rule out issues
4. **Check logs**: `docker-compose logs worker --tail=100`

---

**Remember**: The ChromaDB log message "Collection kinetic_vectors is not created" is **normal** - it just means the collection will be created on first use. The real issue was the invalid API key preventing document embedding.
