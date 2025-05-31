# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the OpenEvolve autonomous development pipeline in production environments. It covers Docker, Kubernetes, and monitoring setup.

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB RAM, recommended 16GB+ RAM
- **Storage**: Minimum 50GB, recommended 100GB+ SSD
- **Network**: Stable internet connection for LLM API calls

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for Kubernetes deployment)
- kubectl configured for your cluster
- Helm 3.0+ (optional, for Kubernetes deployment)

## Environment Configuration

### 1. Environment Variables

Create a `.env` file with the following variables:

```bash
# LLM Configuration
LITELLM_DEFAULT_MODEL=gpt-3.5-turbo
FLASH_API_KEY=your_api_key_here
EVALUATION_API_KEY=your_evaluation_api_key_here

# Database Configuration
DB_PASSWORD=secure_database_password
DATABASE_URL=postgresql://openevolve:${DB_PASSWORD}@postgres:5432/openevolve

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your_secret_key_here
GRAFANA_PASSWORD=secure_grafana_password

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Performance Tuning
POPULATION_SIZE=10
GENERATIONS=5
EVALUATION_TIMEOUT_SECONDS=300
```

### 2. SSL Certificates

For production deployment, obtain SSL certificates:

```bash
# Using Let's Encrypt with Certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to deployment directory
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem deployment/docker/ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem deployment/docker/ssl/
```

## Docker Deployment

### 1. Build and Deploy

```bash
# Clone the repository
git clone https://github.com/Zeeeepa/OpenAlpha_Evolve.git
cd OpenAlpha_Evolve

# Build the Docker image
docker build -f deployment/docker/Dockerfile -t openevolve:latest .

# Deploy using Docker Compose
cd deployment/docker
docker-compose up -d
```

### 2. Verify Deployment

```bash
# Check container status
docker-compose ps

# Check application health
curl http://localhost:8000/health

# Check logs
docker-compose logs -f openevolve-app
```

### 3. Scaling Services

```bash
# Scale application instances
docker-compose up -d --scale openevolve-app=3

# Monitor resource usage
docker stats
```

## Kubernetes Deployment

### 1. Prepare Kubernetes Cluster

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Create secrets
kubectl create secret generic openevolve-secrets \
  --from-literal=database-url="postgresql://openevolve:${DB_PASSWORD}@postgres:5432/openevolve" \
  --from-literal=api-key="${FLASH_API_KEY}" \
  -n openevolve

# Create config map
kubectl create configmap openevolve-config \
  --from-literal=default-model="gpt-3.5-turbo" \
  --from-literal=log-level="INFO" \
  -n openevolve
```

### 2. Deploy Database and Redis

```bash
# Deploy PostgreSQL
kubectl apply -f deployment/kubernetes/postgres.yaml

# Deploy Redis
kubectl apply -f deployment/kubernetes/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n openevolve --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n openevolve --timeout=300s
```

### 3. Deploy Application

```bash
# Deploy the main application
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check deployment status
kubectl get deployments -n openevolve
kubectl get pods -n openevolve

# Check application logs
kubectl logs -f deployment/openevolve-app -n openevolve
```

### 4. Configure Ingress

```bash
# Install NGINX Ingress Controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Apply ingress configuration
kubectl apply -f deployment/kubernetes/ingress.yaml

# Check ingress status
kubectl get ingress -n openevolve
```

## Monitoring Setup

### 1. Prometheus and Grafana

```bash
# Deploy monitoring stack
kubectl apply -f deployment/kubernetes/monitoring/

# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n openevolve

# Login to Grafana (admin / your_grafana_password)
# Import dashboards from deployment/monitoring/grafana/dashboards/
```

### 2. Configure Alerts

```bash
# Apply alert rules
kubectl apply -f deployment/monitoring/alert-rules.yaml

# Configure notification channels in Grafana
# - Slack
# - Email
# - PagerDuty
```

### 3. Log Aggregation

```bash
# Deploy ELK stack (optional)
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n openevolve
helm install kibana elastic/kibana -n openevolve
helm install filebeat elastic/filebeat -n openevolve
```

## Performance Tuning

### 1. Resource Allocation

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### 2. Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

### 3. Application Tuning

```bash
# Environment variables for performance
POPULATION_SIZE=20
GENERATIONS=10
EVALUATION_TIMEOUT_SECONDS=600
API_MAX_RETRIES=3
API_RETRY_DELAY_SECONDS=5
```

## Security Configuration

### 1. Network Security

```bash
# Configure firewall rules
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block direct app access
ufw enable
```

### 2. Container Security

```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Read-only filesystem
docker run --read-only --tmpfs /tmp openevolve:latest
```

### 3. Secrets Management

```bash
# Use Kubernetes secrets for sensitive data
kubectl create secret generic api-keys \
  --from-literal=openai-key="${OPENAI_API_KEY}" \
  --from-literal=anthropic-key="${ANTHROPIC_API_KEY}" \
  -n openevolve

# Rotate secrets regularly
kubectl delete secret api-keys -n openevolve
kubectl create secret generic api-keys \
  --from-literal=openai-key="${NEW_OPENAI_API_KEY}" \
  -n openevolve
```

## Backup and Recovery

### 1. Database Backup

```bash
# Automated PostgreSQL backup
kubectl create cronjob postgres-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- /bin/sh -c "pg_dump -h postgres -U openevolve openevolve > /backup/backup-$(date +%Y%m%d).sql"
```

### 2. Application State Backup

```bash
# Backup persistent volumes
kubectl get pv
kubectl create -f backup-job.yaml
```

### 3. Disaster Recovery

```bash
# Restore from backup
kubectl apply -f restore-job.yaml

# Verify data integrity
kubectl exec -it postgres-pod -- psql -U openevolve -c "SELECT COUNT(*) FROM programs;"
```

## Maintenance Procedures

### 1. Rolling Updates

```bash
# Update application image
kubectl set image deployment/openevolve-app openevolve-app=openevolve:v2.0.0 -n openevolve

# Monitor rollout
kubectl rollout status deployment/openevolve-app -n openevolve

# Rollback if needed
kubectl rollout undo deployment/openevolve-app -n openevolve
```

### 2. Health Checks

```bash
# Check system health
curl https://your-domain.com/health

# Check component health
curl https://your-domain.com/health/detailed

# Monitor metrics
curl https://your-domain.com/metrics
```

### 3. Log Management

```bash
# Rotate logs
kubectl exec -it openevolve-app-pod -- logrotate /etc/logrotate.conf

# Clean old logs
find /var/log -name "*.log" -mtime +30 -delete
```

## Troubleshooting

### 1. Common Issues

**Application won't start:**
```bash
# Check logs
kubectl logs deployment/openevolve-app -n openevolve

# Check resource constraints
kubectl describe pod openevolve-app-xxx -n openevolve

# Check secrets and config
kubectl get secrets -n openevolve
kubectl get configmaps -n openevolve
```

**Database connection issues:**
```bash
# Test database connectivity
kubectl exec -it postgres-pod -- psql -U openevolve -c "SELECT 1;"

# Check network policies
kubectl get networkpolicies -n openevolve
```

**Performance issues:**
```bash
# Check resource usage
kubectl top pods -n openevolve
kubectl top nodes

# Check application metrics
curl https://your-domain.com/metrics | grep -E "(cpu|memory|response_time)"
```

### 2. Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/openevolve-app LOG_LEVEL=DEBUG -n openevolve

# Access debug endpoints
curl https://your-domain.com/debug/status
```

## Monitoring and Alerting

### 1. Key Metrics to Monitor

- **Application Metrics:**
  - Request rate and latency
  - Error rate
  - Pipeline execution time
  - Queue depth

- **System Metrics:**
  - CPU and memory usage
  - Disk I/O and space
  - Network traffic
  - Database performance

- **Business Metrics:**
  - Code generation success rate
  - Evaluation accuracy
  - System uptime
  - User satisfaction

### 2. Alert Configuration

```yaml
# Example alert rules
groups:
- name: openevolve-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected

  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    for: 1m
    annotations:
      summary: Database is down
```

## Conclusion

This deployment guide provides a comprehensive approach to running OpenEvolve in production. Regular monitoring, maintenance, and security updates are essential for optimal performance and reliability.

For additional support, refer to the troubleshooting section or contact the development team.

