# FAIVOR-ML-Validator Deployment Guide

## Overview

The FAIVOR-ML-Validator service requires access to Docker to execute ML model containers. This guide explains how to properly deploy the service with Docker access.

## Docker-in-Docker Requirement

The FAIVOR-ML-Validator runs inside a Docker container but needs to spawn other Docker containers for ML model validation. This requires special configuration.

### Why Docker Access is Needed

1. The service validates ML models that are packaged as Docker containers
2. For each validation request, it:
   - Pulls the model's Docker image
   - Starts a temporary container
   - Sends prediction requests to the model
   - Collects results and calculates metrics
   - Stops and removes the container

## Deployment Options

### Option 1: Docker Socket Mounting (Recommended for Development)

Mount the Docker socket from the host into the container:

```bash
docker run -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  faivor-ml-validator
```

**With docker-compose:**
```yaml
services:
  faivor-backend:
    image: faivor-ml-validator
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

**Security Warning:** Mounting the Docker socket gives the container full access to the host's Docker daemon. Use this approach only in trusted environments.

### Option 2: Docker-in-Docker (DinD)

Run Docker daemon inside the container:

```bash
docker run -p 8000:8000 \
  --privileged \
  -e DOCKER_TLS_CERTDIR=/certs \
  -v docker-certs-client:/certs/client:ro \
  faivor-ml-validator
```

### Option 3: Remote Docker API

Connect to a remote Docker daemon:

```bash
docker run -p 8000:8000 \
  -e DOCKER_HOST=tcp://docker-host:2375 \
  faivor-ml-validator
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_HOST` | Docker daemon endpoint | `unix:///var/run/docker.sock` |
| `DOCKER_TLS_VERIFY` | Enable TLS verification | `0` |
| `DOCKER_CERT_PATH` | Path to Docker certificates | `/certs` |
| `CONTAINER_STARTUP_TIMEOUT` | Container startup timeout (seconds) | `60` |
| `DOCKER_EXECUTION_TIMEOUT` | Model execution timeout (seconds) | `360` |

## Security Considerations

### Docker Socket Access
- **Risk:** Container has full access to host Docker daemon
- **Mitigation:** Use only in development or trusted environments
- **Alternative:** Use Docker API with TLS authentication

### Network Isolation
- Model containers run on random ports
- Consider using Docker networks for isolation
- Implement resource limits for model containers

## Troubleshooting

### Error: "Docker is not available or not running"

**Cause:** The service cannot access the Docker daemon.

**Solutions:**
1. Ensure Docker socket is mounted: `-v /var/run/docker.sock:/var/run/docker.sock`
2. Check Docker socket permissions
3. Verify Docker daemon is running on the host

### Error: "Permission denied while trying to connect to Docker daemon"

**Cause:** Container user doesn't have permissions to access Docker socket.

**Solutions:**
1. Run container with appropriate user/group
2. Adjust Docker socket permissions on host
3. Use Docker group mapping

### Error: "Cannot connect to Docker daemon at unix:///var/run/docker.sock"

**Cause:** Docker socket not available at expected location.

**Solutions:**
1. Verify Docker installation on host
2. Check if Docker daemon is running: `systemctl status docker`
3. Ensure socket path is correct for your OS

## Production Deployment

For production environments, consider:

1. **Kubernetes:** Deploy as a Kubernetes Job with appropriate RBAC
2. **Docker Swarm:** Use service constraints and secrets
3. **Cloud Services:** Use managed container services with proper IAM

### Example Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: faivor-ml-validator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: faivor-ml-validator
  template:
    metadata:
      labels:
        app: faivor-ml-validator
    spec:
      containers:
      - name: faivor-ml-validator
        image: faivor-ml-validator:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: docker-sock
          mountPath: /var/run/docker.sock
      volumes:
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
          type: Socket
```

## Monitoring

Monitor the following:
- Docker daemon availability
- Container spawn/cleanup metrics
- Resource usage by model containers
- Failed model executions

## Support

For deployment issues:
1. Check container logs: `docker logs <container-id>`
2. Verify Docker access: `docker exec <container-id> docker ps`
3. Review model container logs for execution errors