# Deployment Scripts and Documentation

## Overview

The `deployment/` folder contains all the necessary configuration files and scripts for deploying the Ollama book examples in production environments. This includes Docker, Kubernetes, monitoring, and infrastructure configurations.

## Contents

### Docker Deployment
- `docker-compose.production.yml` - Production Docker Compose configuration
- `Dockerfile.webapp` - Web application container
- `Dockerfile.api` - API server container
- `.env.production` - Production environment variables

### Nginx Configuration
- `nginx/nginx.conf` - Main Nginx configuration
- `nginx/default.conf` - Server block configuration with reverse proxy

### Kubernetes Deployment
- `k8s/ollama-deployment.yaml` - Ollama server Kubernetes deployment
- `k8s/webapp-deployment.yaml` - Web application deployment
- `k8s/ingress.yaml` - Ingress controller and SSL configuration

### Monitoring
- `monitoring/prometheus.yml` - Prometheus configuration
- `monitoring/alert_rules.yml` - Alerting rules

## Quick Start

### Docker Compose Deployment

1. **Prepare environment:**
   ```bash
   cp .env.production .env
   # Edit .env with your configuration
   ```

2. **Start services:**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

3. **Access applications:**
   - Web UI: http://localhost/web
   - API: http://localhost/api
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

### Kubernetes Deployment

1. **Create namespace:**
   ```bash
   kubectl create namespace ollama
   ```

2. **Deploy components:**
   ```bash
   kubectl apply -f k8s/ -n ollama
   ```

3. **Check status:**
   ```bash
   kubectl get pods -n ollama
   ```

## Services Included

### Core Services
- **Ollama Server**: The main LLM server
- **Web Application**: Flask-based chat interface
- **API Server**: REST API endpoints
- **Nginx**: Reverse proxy and load balancer

### Supporting Services
- **PostgreSQL**: Database for persistent data
- **Redis**: Caching and session storage
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards

## Security Features

### Authentication & Authorization
- JWT token-based authentication
- API rate limiting
- CORS configuration
- Input validation

### SSL/TLS
- HTTPS redirect configuration
- SSL certificate management
- Secure headers

### Container Security
- Non-root user containers
- Resource limits
- Health checks
- Security scanning

## Monitoring & Observability

### Metrics
- Application performance metrics
- Infrastructure metrics
- Custom business metrics
- Real-time dashboards

### Alerting
- Service availability alerts
- Performance threshold alerts
- Resource usage alerts
- Error rate monitoring

### Logging
- Centralized logging
- Structured JSON logs
- Log rotation
- Error tracking

## Production Considerations

### Scalability
- Horizontal pod autoscaling
- Load balancing
- Database connection pooling
- Caching strategies

### High Availability
- Multi-replica deployments
- Health checks and restarts
- Rolling updates
- Backup strategies

### Performance
- Resource optimization
- Connection pooling
- Caching layers
- CDN integration

## Environment-Specific Configurations

### Development
```bash
docker-compose up -d
```

### Staging
```bash
docker-compose -f docker-compose.production.yml up -d
```

### Production
```bash
# With monitoring and SSL
docker-compose -f docker-compose.production.yml -f docker-compose.monitoring.yml up -d
```

## Troubleshooting

### Common Issues

1. **Ollama not responding:**
   ```bash
   docker logs ollama-production
   kubectl logs deployment/ollama-server -n ollama
   ```

2. **Web app connection issues:**
   ```bash
   # Check network connectivity
   docker exec -it ollama-webapp curl http://ollama:11434/api/tags
   ```

3. **High memory usage:**
   ```bash
   # Monitor resources
   docker stats
   kubectl top pods -n ollama
   ```

### Debug Mode
```bash
# Enable debug logging
docker-compose -f docker-compose.production.yml exec web-app tail -f /app/logs/app.log
```

## Backup and Recovery

### Database Backup
```bash
docker exec ollama-postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql
```

### Model Data Backup
```bash
docker run --rm -v ollama_models:/data -v $(pwd):/backup alpine tar czf /backup/ollama-models.tar.gz /data
```

## Updates and Maintenance

### Rolling Updates
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d

# Kubernetes
kubectl set image deployment/ollama-webapp webapp=ollama-webapp:v2.0.0 -n ollama
```

### Health Checks
```bash
# Check all services
curl http://localhost/health
curl http://localhost/api/health
curl http://localhost/ollama/api/tags
```

## Resource Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Disk: 100GB SSD
- Network: 1Gbps

### Recommended Production
- CPU: 8+ cores
- RAM: 32GB+
- Disk: 500GB+ NVMe SSD
- Network: 10Gbps
- GPU: Optional (for larger models)

## Support and Maintenance

For support and maintenance:
1. Check application logs
2. Review monitoring dashboards
3. Consult troubleshooting guides
4. Contact support team

---

**Note**: Always test deployments in a staging environment before production deployment.
