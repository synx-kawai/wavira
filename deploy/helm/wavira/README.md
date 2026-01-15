# Wavira Helm Chart

A Helm chart for deploying Wavira - Wi-Fi CSI based person re-identification and crowd estimation system.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.0+
- NGINX Ingress Controller (optional)
- Storage class for persistent volumes

## Installation

```bash
# Add the repository (if published)
# helm repo add wavira https://charts.example.com/wavira
# helm repo update

# Install from local directory
helm install wavira deploy/helm/wavira/

# Install with custom values
helm install wavira deploy/helm/wavira/ -f my-values.yaml

# Install in a specific namespace
helm install wavira deploy/helm/wavira/ -n wavira --create-namespace
```

## Configuration

See `values.yaml` for the full list of configurable parameters.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `historyCollector.enabled` | Enable History Collector | `true` |
| `historyCollector.replicaCount` | Number of replicas | `1` |
| `historyCollector.image.tag` | Image tag | `latest` |
| `mosquitto.enabled` | Deploy internal MQTT broker | `true` |
| `mosquitto.external` | Use external MQTT broker | `false` |
| `ingress.enabled` | Enable Ingress | `true` |
| `security.apiKey` | API key for authentication | `""` |
| `security.requireApiKey` | Require API key | `false` |

### Using External MQTT Broker

```yaml
# values-external-mqtt.yaml
mosquitto:
  enabled: true
  external: true
  externalHost: mqtt.example.com
  externalPort: 1883
```

### Production Configuration

```yaml
# values-production.yaml
historyCollector:
  replicaCount: 2
  resources:
    requests:
      memory: "256Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

security:
  apiKey: "your-secure-api-key"
  requireApiKey: true

ingress:
  api:
    host: wavira.example.com
    tls:
      enabled: true
      secretName: wavira-tls
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade wavira deploy/helm/wavira/ -f my-values.yaml

# Upgrade to a specific version
helm upgrade wavira deploy/helm/wavira/ --version 0.2.0
```

## Uninstallation

```bash
# Uninstall the release
helm uninstall wavira

# Uninstall and delete PVCs
helm uninstall wavira
kubectl delete pvc -l app.kubernetes.io/instance=wavira
```

## Template Rendering

```bash
# Render templates locally
helm template wavira deploy/helm/wavira/

# Render with custom values
helm template wavira deploy/helm/wavira/ -f my-values.yaml

# Debug template rendering
helm template wavira deploy/helm/wavira/ --debug
```

## Chart Development

```bash
# Lint the chart
helm lint deploy/helm/wavira/

# Package the chart
helm package deploy/helm/wavira/

# Test installation (dry-run)
helm install wavira deploy/helm/wavira/ --dry-run --debug
```

## Support

For issues and feature requests, please visit:
https://github.com/your-org/wavira/issues
