#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log) 2>&1
yum update -y
yum install -y docker python3 python3-pip nginx git
systemctl start docker && systemctl enable docker
usermod -aG docker ec2-user
mkdir -p /opt/wavira/{services,data,mosquitto/config,mosquitto/data,dashboard}
cd /opt/wavira
python3 -m venv venv && source venv/bin/activate && pip install paho-mqtt fastapi uvicorn numpy
cat>/opt/wavira/mosquitto/config/mosquitto.conf<<'E'
listener 1883 0.0.0.0
listener 9001 0.0.0.0
protocol websockets
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
E
docker run -d --name mosquitto --restart unless-stopped -p 1883:1883 -p 9001:9001 -v /opt/wavira/mosquitto/config:/mosquitto/config -v /opt/wavira/mosquitto/data:/mosquitto/data eclipse-mosquitto:2
cat>/opt/wavira/services/proc.py<<'P'
import json,time,os
from collections import deque
import numpy as np
import paho.mqtt.client as mqtt
D={}
def on_c(c,u,f,r,p):c.subscribe("wavira/csi/#")if r==0 else None
def on_m(c,u,m):
 try:
  did=m.topic.split("/")[-1];d=json.loads(m.payload);cd=d.get("data",[]);rs=d.get("rssi",-100)
  if did not in D:D[did]=deque(maxlen=100)
  if not cd:return
  a=[np.sqrt(cd[i]**2+(cd[i+1]if i+1<len(cd)else 0)**2)for i in range(0,len(cd)-1,2)]
  if not a:return
  a=np.array(a);av=float(np.mean(a));vr=float(np.var(a));D[did].append(av)
  r={"device_id":did,"timestamp":time.time(),"rssi":rs,"amps":a[:64].tolist(),"avg_amplitude":round(av,2),"variance":round(vr,2),"present":vr>50,"breath_ratio":min(vr/500,1)}
  c.publish(f"wavira/analysis/{did}",json.dumps(r))
 except:pass
c=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.on_connect=on_c;c.on_message=on_m
c.connect(os.environ.get("MQTT_HOST","localhost"),1883,60)
c.loop_forever()
P
cat>/opt/wavira/services/api.py<<'A'
import json,time,os,sqlite3,threading
import paho.mqtt.client as mqtt
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
L=threading.Lock();P=os.environ.get("DB_PATH","/opt/wavira/data/h.db")
def gc():return sqlite3.connect(P,check_same_thread=False)
with L:c=gc();c.execute('CREATE TABLE IF NOT EXISTS h(id INTEGER PRIMARY KEY,did TEXT,ts REAL,rssi INT,amp REAL,var REAL,pres INT)');c.execute('CREATE TABLE IF NOT EXISTS d(did TEXT PRIMARY KEY,ls REAL,rssi INT)');c.execute('CREATE INDEX IF NOT EXISTS ix ON h(did,ts)');c.commit();c.close()
def add(did,ts,rs,amp,var,pres):
 with L:c=gc();c.execute('INSERT INTO h(did,ts,rssi,amp,var,pres)VALUES(?,?,?,?,?,?)',(did,ts,rs,amp,var,1 if pres else 0));c.execute('INSERT INTO d(did,ls,rssi)VALUES(?,?,?)ON CONFLICT(did)DO UPDATE SET ls=excluded.ls,rssi=excluded.rssi',(did,ts,rs));c.commit();c.close()
def on_c(cl,u,f,r,p):cl.subscribe("wavira/analysis/#")if r==0 else None
def on_m(cl,u,m):
 try:
  if m.topic.endswith('/presence'):return
  d=json.loads(m.payload);did=d.get("device_id")
  if did:add(did,d.get("timestamp",time.time()),d.get("rssi",-100),d.get("avg_amplitude",0),d.get("variance",0),d.get("present",False))
 except:pass
mc=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2);mc.on_connect=on_c;mc.on_message=on_m
threading.Thread(target=lambda:mc.connect(os.environ.get("MQTT_HOST","localhost"),1883,60)or mc.loop_forever(),daemon=True).start()
app=FastAPI();app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
@app.get("/api/v1/devices")
def gd():
 with L:c=gc();r=c.execute('SELECT did,ls,rssi FROM d ORDER BY ls DESC').fetchall();c.close();n=time.time();return[{"device_id":x[0],"last_seen":x[1],"rssi":x[2],"online":n-x[1]<60 if x[1]else False}for x in r]
@app.get("/api/v1/history/{did}")
def gh(did:str,limit:int=1000):
 with L:c=gc();r=c.execute('SELECT ts,rssi,amp,var,pres FROM h WHERE did=? ORDER BY ts DESC LIMIT ?',(did,limit)).fetchall();c.close();return[{"timestamp":x[0],"rssi":x[1],"avg_amplitude":x[2],"variance":x[3],"present":bool(x[4])}for x in reversed(r)]
@app.get("/api/v1/hourly")
def hr(hours:int=24):
 with L:c=gc();s=time.time()-hours*3600;r=c.execute("SELECT strftime('%Y-%m-%d %H:00',datetime(ts,'unixepoch'))as hr,did,AVG(amp),AVG(var),SUM(pres)*100.0/COUNT(*),COUNT(*)FROM h WHERE ts>? GROUP BY hr,did ORDER BY hr DESC",(s,)).fetchall();c.close();return[{"hour":x[0],"device_id":x[1],"avg_amplitude":round(x[2],2)if x[2]else 0,"presence_pct":round(x[4],1)if x[4]else 0}for x in r]
if __name__=="__main__":uvicorn.run(app,host="0.0.0.0",port=8080)
A
# Download dashboard from S3
aws s3 cp s3://${s3_bucket}/dashboard/index.html /opt/wavira/dashboard/index.html --region ${aws_region}
cat>/etc/systemd/system/wavira-proc.service<<E
[Unit]
After=network.target docker.service
[Service]
User=ec2-user
WorkingDirectory=/opt/wavira
ExecStart=/opt/wavira/venv/bin/python /opt/wavira/services/proc.py
Restart=always
Environment=PYTHONUNBUFFERED=1 MQTT_HOST=localhost
[Install]
WantedBy=multi-user.target
E
cat>/etc/systemd/system/wavira-api.service<<E
[Unit]
After=network.target docker.service
[Service]
User=ec2-user
WorkingDirectory=/opt/wavira
ExecStart=/opt/wavira/venv/bin/python /opt/wavira/services/api.py
Restart=always
Environment=PYTHONUNBUFFERED=1 MQTT_HOST=localhost DB_PATH=/opt/wavira/data/h.db
[Install]
WantedBy=multi-user.target
E
cat>/etc/nginx/conf.d/wavira.conf<<'N'
server {
    listen 80;
    root /opt/wavira/dashboard;
    index index.html;
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    location /mqtt {
        proxy_pass http://127.0.0.1:9001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    location / {
        try_files $uri $uri/ /index.html;
    }
}
N
chown -R ec2-user:ec2-user /opt/wavira
systemctl daemon-reload && systemctl enable nginx wavira-proc wavira-api && systemctl start nginx
sleep 5 && systemctl start wavira-proc wavira-api

# Install and configure Cloudflare Tunnel
curl -L --output cloudflared.rpm https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm
yum localinstall -y cloudflared.rpm
sudo cloudflared service install eyJhIjoiZDM4MDUyMTJkODU1OTM1NzgwMTMzZWFjMjRkNWMwNzUiLCJ0IjoiNTYxMDgyY2QtNzllYi00NGRiLWI5MjgtZDI3OTg4NGQzYTM1IiwicyI6Ik9HUmhaV0U1TWprdE9USXdPUzAwWVRKbUxUZzBNRGt0WTJaaU1qVTJPRFUxTkRndyJ9

