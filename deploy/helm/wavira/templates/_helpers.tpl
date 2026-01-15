{{/*
Expand the name of the chart.
*/}}
{{- define "wavira.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "wavira.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "wavira.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "wavira.labels" -}}
helm.sh/chart: {{ include "wavira.chart" . }}
{{ include "wavira.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "wavira.selectorLabels" -}}
app.kubernetes.io/name: {{ include "wavira.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
History Collector labels
*/}}
{{- define "wavira.historyCollector.labels" -}}
{{ include "wavira.labels" . }}
app.kubernetes.io/component: history-collector
{{- end }}

{{/*
History Collector selector labels
*/}}
{{- define "wavira.historyCollector.selectorLabels" -}}
{{ include "wavira.selectorLabels" . }}
app.kubernetes.io/component: history-collector
{{- end }}

{{/*
Mosquitto labels
*/}}
{{- define "wavira.mosquitto.labels" -}}
{{ include "wavira.labels" . }}
app.kubernetes.io/component: mosquitto
{{- end }}

{{/*
Mosquitto selector labels
*/}}
{{- define "wavira.mosquitto.selectorLabels" -}}
{{ include "wavira.selectorLabels" . }}
app.kubernetes.io/component: mosquitto
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "wavira.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "wavira.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
MQTT Host - returns internal service name or external host
*/}}
{{- define "wavira.mqttHost" -}}
{{- if .Values.mosquitto.external }}
{{- .Values.mosquitto.externalHost }}
{{- else }}
{{- printf "%s-mosquitto" (include "wavira.fullname" .) }}
{{- end }}
{{- end }}

{{/*
MQTT Port - returns internal port or external port
*/}}
{{- define "wavira.mqttPort" -}}
{{- if .Values.mosquitto.external }}
{{- .Values.mosquitto.externalPort }}
{{- else }}
{{- .Values.mosquitto.service.mqttPort }}
{{- end }}
{{- end }}
