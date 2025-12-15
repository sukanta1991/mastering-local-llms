"""
Docker Deployment Examples for Ollama
Chapter 18: Docker Deployment
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time


class DockerOllamaManager:
    """Manage Ollama deployment using Docker."""
    
    def __init__(self, container_name: str = "ollama-server"):
        self.container_name = container_name
        self.logger = logging.getLogger(__name__)
        self.ollama_port = 11434
        self.host_port = 11434
    
    def check_docker_installed(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(f"Docker version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Docker is not installed or not running")
            return False
    
    def pull_ollama_image(self, tag: str = "latest") -> bool:
        """Pull the Ollama Docker image."""
        try:
            self.logger.info(f"Pulling Ollama image: ollama/ollama:{tag}")
            result = subprocess.run(
                ["docker", "pull", f"ollama/ollama:{tag}"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Successfully pulled Ollama image")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull Ollama image: {e}")
            return False
    
    def is_container_running(self) -> bool:
        """Check if the Ollama container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.container_name in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def start_container(
        self,
        gpu_support: bool = False,
        volume_path: Optional[str] = None,
        environment_vars: Optional[Dict[str, str]] = None
    ) -> bool:
        """Start the Ollama container."""
        try:
            # Stop existing container if running
            if self.is_container_running():
                self.logger.info("Stopping existing container")
                self.stop_container()
            
            # Remove existing container if it exists
            self.remove_container()
            
            # Build docker run command
            cmd = ["docker", "run", "-d"]
            
            # Add GPU support if requested
            if gpu_support:
                cmd.extend(["--gpus", "all"])
            
            # Add port mapping
            cmd.extend(["-p", f"{self.host_port}:{self.ollama_port}"])
            
            # Add volume mapping for model storage
            if volume_path:
                volume_path = os.path.abspath(volume_path)
                os.makedirs(volume_path, exist_ok=True)
                cmd.extend(["-v", f"{volume_path}:/root/.ollama"])
            
            # Add environment variables
            if environment_vars:
                for key, value in environment_vars.items():
                    cmd.extend(["-e", f"{key}={value}"])
            
            # Add container name and image
            cmd.extend(["--name", self.container_name, "ollama/ollama"])
            
            self.logger.info(f"Starting container with command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            
            self.logger.info(f"Container started with ID: {container_id}")
            
            # Wait for container to be ready
            self.wait_for_container_ready()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container: {e}")
            return False
    
    def stop_container(self) -> bool:
        """Stop the Ollama container."""
        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Container stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop container: {e}")
            return False
    
    def remove_container(self) -> bool:
        """Remove the Ollama container."""
        try:
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def wait_for_container_ready(self, timeout: int = 60) -> bool:
        """Wait for the container to be ready to accept requests."""
        self.logger.info("Waiting for container to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if we can connect to the API
                result = subprocess.run(
                    ["docker", "exec", self.container_name, "ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.logger.info("Container is ready")
                    return True
            except subprocess.TimeoutExpired:
                pass
            
            time.sleep(2)
        
        self.logger.error("Container failed to become ready within timeout")
        return False
    
    def execute_ollama_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Execute an Ollama command inside the container."""
        full_command = ["docker", "exec", self.container_name, "ollama"] + command
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(full_command)}")
            self.logger.error(f"Error: {e}")
            raise
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model inside the container."""
        try:
            self.logger.info(f"Pulling model: {model_name}")
            result = self.execute_ollama_command(["pull", model_name])
            self.logger.info(f"Successfully pulled model: {model_name}")
            return True
        except subprocess.CalledProcessError:
            self.logger.error(f"Failed to pull model: {model_name}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in the container."""
        try:
            result = self.execute_ollama_command(["list"])
            
            # Parse the output to extract model names
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            
            return models
        except subprocess.CalledProcessError:
            self.logger.error("Failed to list models")
            return []
    
    def get_container_logs(self, lines: int = 50) -> str:
        """Get container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), self.container_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get container logs: {e}")
            return ""
    
    def get_container_stats(self) -> Dict:
        """Get container resource usage statistics."""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", 
                 "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}",
                 self.container_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # Parse the stats line
                parts = lines[1].split('\t')
                if len(parts) >= 6:
                    return {
                        "container": parts[0],
                        "cpu_percent": parts[1],
                        "memory_usage": parts[2],
                        "memory_percent": parts[3],
                        "network_io": parts[4],
                        "block_io": parts[5]
                    }
            
            return {}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get container stats: {e}")
            return {}


class DockerComposeManager:
    """Manage Ollama deployment using Docker Compose."""
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = compose_file
        self.logger = logging.getLogger(__name__)
    
    def create_compose_file(
        self,
        models_to_preload: List[str] = None,
        gpu_support: bool = False,
        environment_vars: Dict[str, str] = None
    ):
        """Create a Docker Compose file for Ollama."""
        
        compose_content = {
            "version": "3.8",
            "services": {
                "ollama": {
                    "image": "ollama/ollama:latest",
                    "container_name": "ollama-server",
                    "restart": "unless-stopped",
                    "ports": ["11434:11434"],
                    "volumes": ["ollama_data:/root/.ollama"],
                    "environment": environment_vars or {}
                }
            },
            "volumes": {
                "ollama_data": {
                    "driver": "local"
                }
            }
        }
        
        # Add GPU support if requested
        if gpu_support:
            compose_content["services"]["ollama"]["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "count": "all",
                            "capabilities": ["gpu"]
                        }]
                    }
                }
            }
        
        # Add model initialization service if models are specified
        if models_to_preload:
            init_commands = []
            for model in models_to_preload:
                init_commands.append(f"ollama pull {model}")
            
            compose_content["services"]["ollama-init"] = {
                "image": "ollama/ollama:latest",
                "container_name": "ollama-init",
                "depends_on": ["ollama"],
                "volumes": ["ollama_data:/root/.ollama"],
                "entrypoint": ["/bin/bash", "-c"],
                "command": [
                    f"sleep 10 && {' && '.join(init_commands)}"
                ],
                "restart": "no"
            }
        
        # Write compose file
        with open(self.compose_file, 'w') as f:
            import yaml
            yaml.dump(compose_content, f, default_flow_style=False)
        
        self.logger.info(f"Created Docker Compose file: {self.compose_file}")
    
    def start_services(self) -> bool:
        """Start services using Docker Compose."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "up", "-d"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Services started successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start services: {e}")
            return False
    
    def stop_services(self) -> bool:
        """Stop services using Docker Compose."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "down"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Services stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop services: {e}")
            return False


def create_production_dockerfile():
    """Create a production-ready Dockerfile for Ollama."""
    dockerfile_content = """# Production Ollama Dockerfile
FROM ollama/ollama:latest

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434

# Create non-root user for security
RUN groupadd -r ollama && useradd -r -g ollama -d /home/ollama -s /bin/bash ollama

# Create directories and set permissions
RUN mkdir -p /home/ollama/.ollama && \
    chown -R ollama:ollama /home/ollama

# Switch to non-root user
USER ollama
WORKDIR /home/ollama

# Copy any custom configuration files
# COPY --chown=ollama:ollama config/ /home/ollama/.ollama/

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ollama list || exit 1

# Start Ollama
CMD ["ollama", "serve"]
"""
    
    with open("Dockerfile.production", "w") as f:
        f.write(dockerfile_content)
    
    print("Created production Dockerfile: Dockerfile.production")


def create_kubernetes_manifests():
    """Create Kubernetes manifests for Ollama deployment."""
    
    # Deployment manifest
    deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-deployment
  labels:
    app: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        env:
        - name: OLLAMA_HOST
          value: "0.0.0.0"
        - name: OLLAMA_PORT
          value: "11434"
        volumeMounts:
        - name: ollama-data
          mountPath: /root/.ollama
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          exec:
            command:
            - ollama
            - list
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - ollama
            - list
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
spec:
  selector:
    app: ollama
  ports:
  - protocol: TCP
    port: 11434
    targetPort: 11434
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
"""
    
    with open("ollama-k8s.yaml", "w") as f:
        f.write(deployment_yaml)
    
    print("Created Kubernetes manifests: ollama-k8s.yaml")


# Example usage and main function
def main():
    """Main function demonstrating Docker deployment."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Docker manager
    manager = DockerOllamaManager("ollama-demo")
    
    if not manager.check_docker_installed():
        print("Docker is not installed. Please install Docker first.")
        return
    
    # Pull Ollama image
    print("Pulling Ollama Docker image...")
    if not manager.pull_ollama_image():
        print("Failed to pull image")
        return
    
    # Start container with GPU support (if available)
    print("Starting Ollama container...")
    success = manager.start_container(
        gpu_support=False,  # Set to True if you have GPU support
        volume_path="./ollama_data",
        environment_vars={
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_PORT": "11434"
        }
    )
    
    if not success:
        print("Failed to start container")
        return
    
    # Pull a model
    print("Pulling llama2 model...")
    manager.pull_model("llama2")
    
    # List models
    models = manager.list_models()
    print(f"Available models: {models}")
    
    # Show container stats
    stats = manager.get_container_stats()
    if stats:
        print(f"Container stats: {stats}")
    
    # Create additional deployment files
    print("\nCreating additional deployment files...")
    create_production_dockerfile()
    create_kubernetes_manifests()
    
    # Create Docker Compose setup
    compose_manager = DockerComposeManager()
    try:
        compose_manager.create_compose_file(
            models_to_preload=["llama2"],
            gpu_support=False,
            environment_vars={
                "OLLAMA_HOST": "0.0.0.0",
                "OLLAMA_KEEP_ALIVE": "24h"
            }
        )
        print("Created Docker Compose file")
    except ImportError:
        print("PyYAML not installed, skipping Docker Compose file creation")
    
    print("\nOllama is now running in Docker!")
    print(f"API endpoint: http://localhost:{manager.host_port}")
    print("Use 'docker logs ollama-demo' to view logs")
    print("Use 'docker stop ollama-demo' to stop the container")


if __name__ == "__main__":
    main()
