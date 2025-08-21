# Coolify Deployment Guide - Ollama Enterprise News Analysis Service

## Prerequisites

1. **Coolify Server** with Docker and Docker Compose
2. **Ollama Service** running and accessible at `http://20.64.243.4:11434`
3. **Domain/Subdomain** (optional, for custom URL)

## Quick Deploy Steps

### 1. Create New Resource in Coolify

1. Login to your Coolify dashboard
2. Click **"+ New Resource"**
3. Select **"Docker Compose"**
4. Choose your server/destination

### 2. Configure Git Repository

**Repository URL**: `https://github.com/your-username/ollama_service.git`

Or upload the project files directly to Coolify.

### 3. Environment Variables

In Coolify, set these environment variables:

#### Required Variables
```bash
# Ollama Configuration
OLLAMA_PRIMARY_URL=http://20.64.243.4:11434
OLLAMA_FALLBACK_URL=http://172.17.0.1:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=60

# API Security
API_KEY=your-secure-api-key-here

# Application
API_HOST=0.0.0.0
API_PORT=8000
PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
DEBUG=false
```

#### Optional Variables
```bash
# Rate Limiting
RATE_LIMIT_PER_HOUR=1000

# Redis Cache (if using Redis)
# REDIS_URL=redis://redis-service:6379/0
# CACHE_TTL=900

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### 4. Domain Configuration (Optional)

If you want a custom domain:
1. Go to **"Domains"** tab in Coolify
2. Add your domain: `ollama-api.yourdomain.com`
3. Coolify will auto-generate SSL certificates

### 5. Deploy

1. Click **"Deploy"** button
2. Monitor logs in Coolify dashboard
3. Wait for deployment to complete

## Health Check

The application includes built-in health checks:

- **Health Endpoint**: `GET /api/v1/analyze-comprehensive/health`
- **Docker Health Check**: Runs every 30 seconds
- **Coolify Monitoring**: Automatic restart if unhealthy

## Testing Deployment

### 1. Test Health Check
```bash
curl https://your-domain.com/api/v1/analyze-comprehensive/health
```

### 2. Test API Endpoint
```bash
curl -X POST "https://your-domain.com/api/v1/analyze-comprehensive" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tesla reports strong quarterly earnings with stock prices rising 10% after hours.",
    "title": "Tesla Strong Q3 Earnings",
    "publisher": "Tech News"
  }'
```

### 3. Access Swagger UI
Visit: `https://your-domain.com/docs`

## Coolify Features Used

### ✅ Auto SSL Certificates
- Automatic Let's Encrypt SSL
- HTTPS enforced by default

### ✅ Health Monitoring
- Container health checks every 30s
- Auto-restart on failures
- Email/Slack notifications (configurable)

### ✅ Log Management
- Centralized logging in Coolify dashboard
- JSON structured logs for easy parsing
- Log retention and rotation

### ✅ Zero Downtime Deployments
- Rolling updates with health checks
- Automatic rollback on failure

### ✅ Environment Management
- Secure environment variable storage
- Per-environment configurations
- Secret management

## Performance Tuning

### Resource Allocation
```yaml
# In Coolify resource settings:
CPU: 1-2 cores
Memory: 2-4 GB RAM
Storage: 10+ GB
```

### Scaling (if needed)
```yaml
# For high load environments:
replicas: 2-3
load_balancer: coolify_proxy
```

## Troubleshooting

### 1. Ollama Connection Issues
- Check `OLLAMA_PRIMARY_URL` is accessible from Coolify server
- Test: `curl http://20.64.243.4:11434/api/tags` from Coolify server
- Verify firewall rules allow connections

### 2. Build Failures
```bash
# Check build logs in Coolify for:
- Python package installation errors
- Missing dependencies
- Docker image issues
```

### 3. Runtime Errors
```bash
# Check application logs for:
- API authentication failures
- Ollama model availability
- Memory/resource constraints
```

### 4. Health Check Failures
```bash
# Common issues:
- Port binding conflicts (ensure PORT=8000)
- Ollama service unreachable
- Missing health check endpoint
```

## Monitoring & Maintenance

### Log Analysis
```bash
# Key log messages to monitor:
- "✓ Primary Ollama URL is accessible"
- "✓ Model 'llama3.2:1b' is available" 
- "Successfully connected to Ollama service"
- Request processing times (should be <20s)
```

### Performance Metrics
- **Response Time**: <20 seconds average
- **Throughput**: 100+ requests/hour per instance
- **Memory Usage**: <2GB under normal load
- **CPU Usage**: <50% average

### Updates
1. Push code changes to Git repository
2. Coolify auto-deploys on new commits (if configured)
3. Or manually trigger deployment in Coolify dashboard

## Security Considerations

### 🔒 API Key Security
- Use strong, random API keys
- Rotate keys regularly
- Store in Coolify environment variables (encrypted)

### 🔒 Network Security
- Use HTTPS only (enforced by Coolify)
- Restrict access to Ollama service
- Configure rate limiting appropriately

### 🔒 Container Security
- Non-root user in Docker container
- Minimal base image (Python slim)
- Regular dependency updates

## Cost Optimization

### Resource Monitoring
- Use Coolify metrics to right-size resources
- Scale down during off-peak hours
- Monitor container resource usage

### Efficient Processing
- Current setup: ~14-16 seconds per request
- Batch processing available for multiple articles
- Caching with Redis (optional) for repeated requests

## Support

### Service Status
- Health Check: `/api/v1/analyze-comprehensive/health`
- Swagger Docs: `/docs`
- OpenAPI Schema: `/openapi.json`

### Key Endpoints
- **Analysis**: `POST /api/v1/analyze-comprehensive`
- **Batch**: `POST /api/v1/analyze-comprehensive/batch`
- **Health**: `GET /api/v1/analyze-comprehensive/health`

### Contact & Issues
- Check Coolify deployment logs first
- Verify Ollama service connectivity
- Monitor resource usage and scaling needs