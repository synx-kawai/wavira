# Wavira Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Wavira using Kustomize.

## Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured to access your cluster
- NGINX Ingress Controller (optional, for external access)
- Storage class for persistent volumes

## Quick Start

```bash
# Apply all resources
kubectl apply -k deploy/kubernetes/

# Or using kustomize
kustomize build deploy/kubernetes/ | kubectl apply -f -
```

## Components

| File | Description |
|------|-------------|
| `namespace.yaml` | Creates the `wavira` namespace |
| `configmap.yaml` | Non-sensitive configuration (MQTT, API settings) |
| `secret.yaml` | Template for API keys and MQTT passwords |
| `mosquitto-deployment.yaml` | MQTT broker deployment |
| `mosquitto-service.yaml` | MQTT ClusterIP service |
| `history-collector-deployment.yaml` | REST API and data collector |
| `history-collector-service.yaml` | REST API ClusterIP service |
| `ingress.yaml` | External access configuration |
| `kustomization.yaml` | Kustomize configuration |

## Configuration

### Secrets

Before deploying, update `secret.yaml` with your secure values:

```bash
# Generate a secure API key
openssl rand -hex 32
```

For production, consider using:
- Sealed Secrets
- External Secrets Operator
- HashiCorp Vault

### Environment-specific Customization

Create overlay directories for different environments:

```
deploy/kubernetes/
├── base/           # Move current files here
└── overlays/
    ├── development/
    ├── staging/
    └── production/
```

Example production overlay:

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patches:
  - path: patches/increase-replicas.yaml
images:
  - name: wavira/history-collector
    newTag: v1.0.0
```

## Accessing Services

### Port Forwarding

```bash
# REST API
kubectl port-forward -n wavira svc/history-collector 8080:8080

# MQTT Broker
kubectl port-forward -n wavira svc/mosquitto 1883:1883 9001:9001
```

### Using Ingress

1. Update `ingress.yaml` with your domain
2. Configure DNS to point to your ingress controller
3. (Optional) Add TLS certificates

## Health Checks

```bash
# Check deployment status
kubectl get pods -n wavira

# Check service endpoints
kubectl get endpoints -n wavira

# Test health endpoint
kubectl exec -n wavira deploy/history-collector -- curl -s localhost:8080/api/v1/health
```

## Troubleshooting

### Pods not starting

```bash
kubectl describe pod -n wavira <pod-name>
kubectl logs -n wavira <pod-name>
```

### MQTT connection issues

```bash
# Check mosquitto logs
kubectl logs -n wavira -l app.kubernetes.io/component=mosquitto

# Test MQTT connectivity from another pod
kubectl exec -n wavira deploy/history-collector -- nc -zv mosquitto 1883
```

### Database issues

```bash
# Check PVC status
kubectl get pvc -n wavira

# Access history collector shell
kubectl exec -it -n wavira deploy/history-collector -- sh
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k deploy/kubernetes/

# Or delete namespace (removes everything)
kubectl delete namespace wavira
```
