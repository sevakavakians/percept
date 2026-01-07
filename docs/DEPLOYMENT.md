# PERCEPT Deployment Guide

Instructions for deploying PERCEPT on Raspberry Pi 5 with Hailo-8 accelerator.

## Table of Contents

1. [Hardware Setup](#hardware-setup)
2. [Software Installation](#software-installation)
3. [Configuration](#configuration)
4. [Running as a Service](#running-as-a-service)
5. [Production Checklist](#production-checklist)
6. [Monitoring](#monitoring)
7. [Backup and Recovery](#backup-and-recovery)

---

## Hardware Setup

### Required Components

| Component | Specification | Notes |
|-----------|---------------|-------|
| Raspberry Pi 5 | 8GB RAM | 4GB may work with reduced features |
| Hailo-8 | M.2 or HAT form factor | 26 TOPS AI accelerator |
| RealSense Camera | D415 or D455 | D455 has wider FOV |
| Storage | 64GB+ microSD or NVMe | NVMe recommended for database |
| Power | 27W USB-C | Official Pi 5 power supply |
| Cooling | Active cooling | Required for sustained operation |

### Physical Installation

1. **Hailo-8 M.2 Installation:**
   - Power off the Pi
   - Insert M.2 HAT or Pironman5 enclosure
   - Connect M.2 Hailo-8 module
   - Secure with screws

2. **RealSense Camera:**
   - Connect to USB 3.0 port (blue)
   - Avoid USB hubs - connect directly
   - Mount securely to prevent vibration

3. **Cooling:**
   - Install active cooler or fan
   - Ensure adequate airflow
   - Consider Pironman5 enclosure for integrated cooling

### Verify Hardware

```bash
# Check Hailo-8
hailortcli fw-control identify
# Expected: Hailo-8 identified with firmware version

# Check RealSense
rs-enumerate-devices
# Expected: Camera serial number and specs listed

# Check temperature
vcgencmd measure_temp
# Should be under 70°C under load
```

---

## Software Installation

### System Preparation

1. **Update system:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install system dependencies:**
   ```bash
   sudo apt install -y \
       python3-pip python3-venv \
       libgl1-mesa-glx libglib2.0-0 \
       librealsense2-dkms librealsense2-utils \
       git build-essential
   ```

3. **Install Hailo runtime:**
   ```bash
   # Follow Hailo AI Software Suite installation guide
   # https://hailo.ai/developer-zone/

   # Verify installation
   dpkg -l | grep hailo
   ```

### PERCEPT Installation

1. **Clone repository:**
   ```bash
   cd /opt
   sudo git clone https://github.com/yourusername/percept.git
   sudo chown -R $USER:$USER percept
   cd percept
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install PERCEPT:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[production]"
   ```

4. **Download models:**
   ```bash
   # Models should be in /usr/share/hailo-models/
   ls /usr/share/hailo-models/
   # Should include: fastsam_s.hef, yolov8s.hef, etc.
   ```

5. **Initialize database:**
   ```bash
   percept init-db
   ```

### Verify Installation

```bash
# Run tests (without hardware)
pytest tests/ -k "not hailo and not realsense"

# Test with hardware
percept check-hardware
```

---

## Configuration

### Environment Variables

Create `/opt/percept/.env`:

```bash
# Runtime settings
PERCEPT_ENV=production
PERCEPT_DEBUG=0
PERCEPT_LOG_LEVEL=INFO

# Database
PERCEPT_DB_PATH=/var/lib/percept/percept.db

# Web interface
PERCEPT_HOST=0.0.0.0
PERCEPT_PORT=8080

# Hardware
HAILO_DEVICE_ID=0
REALSENSE_SERIAL=auto
```

### Configuration File

Create `/opt/percept/config/production.yaml`:

```yaml
# Production configuration
environment: production

# Cameras
cameras:
  - id: main
    serial: auto  # Auto-detect first camera
    resolution: [640, 480]
    fps: 30
    depth_enabled: true
    auto_exposure: true

# Segmentation
segmentation:
  model: fastsam
  model_path: /usr/share/hailo-models/fastsam_s.hef
  confidence_threshold: 0.5
  use_depth: true
  depth_discontinuity_threshold: 0.3

# Tracking
tracking:
  match_threshold: 0.7
  max_age: 30
  min_hits: 3
  embedding_model: osnet
  gallery_size: 10000

# Adaptive processing
adaptive:
  enabled: true
  target_fps: 15.0
  min_fps: 10.0

# Database
database:
  path: /var/lib/percept/percept.db
  embedding_storage: /var/lib/percept/embeddings
  max_objects: 100000
  cleanup_days: 30

# Logging
logging:
  level: INFO
  file: /var/log/percept/percept.log
  max_size_mb: 100
  backup_count: 5
```

### Directory Setup

```bash
# Create required directories
sudo mkdir -p /var/lib/percept
sudo mkdir -p /var/log/percept
sudo chown -R $USER:$USER /var/lib/percept /var/log/percept
```

---

## Running as a Service

### Systemd Service

Create `/etc/systemd/system/percept.service`:

```ini
[Unit]
Description=PERCEPT Vision Processing
After=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/percept
Environment="PATH=/opt/percept/venv/bin:/usr/local/bin:/usr/bin"
EnvironmentFile=/opt/percept/.env
ExecStart=/opt/percept/venv/bin/percept start --config /opt/percept/config/production.yaml
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/percept /var/log/percept

[Install]
WantedBy=multi-user.target
```

### Enable Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable percept

# Start service
sudo systemctl start percept

# Check status
sudo systemctl status percept

# View logs
journalctl -u percept -f
```

### Web Interface (Nginx)

For production, use Nginx as reverse proxy:

```nginx
# /etc/nginx/sites-available/percept
server {
    listen 80;
    server_name percept.local;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/percept /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Production Checklist

### Before Deployment

- [ ] Hardware verified and stable
- [ ] Hailo-8 firmware up to date
- [ ] RealSense firmware up to date
- [ ] All tests passing
- [ ] Configuration validated
- [ ] Database initialized
- [ ] Logs directory created
- [ ] Service file created
- [ ] Backup strategy in place

### Security

- [ ] Change default passwords
- [ ] Enable firewall (ufw)
- [ ] Restrict SSH access
- [ ] Configure fail2ban
- [ ] Review file permissions
- [ ] Enable HTTPS (if exposed)

### Performance

- [ ] Adequate cooling verified
- [ ] CPU governor set to performance
- [ ] Memory swap disabled (if sufficient RAM)
- [ ] GPU memory split optimized

```bash
# Set CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap (optional, if 8GB RAM)
sudo swapoff -a

# GPU memory (in /boot/config.txt)
gpu_mem=128
```

---

## Monitoring

### Built-in Metrics

Access metrics at `http://localhost:8080/api/metrics`:

```json
{
  "fps": 15.2,
  "cpu_usage": 45.5,
  "memory_usage": 62.3,
  "objects_tracked": 1250,
  "uptime_seconds": 86400
}
```

### Prometheus Integration

Enable Prometheus metrics in config:

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
```

Metrics available at `http://localhost:9090/metrics`.

### Health Checks

```bash
# Simple health check
curl -s http://localhost:8080/health | jq .

# Detailed check script
#!/bin/bash
HEALTH=$(curl -s http://localhost:8080/health)
STATUS=$(echo $HEALTH | jq -r '.status')

if [ "$STATUS" != "healthy" ]; then
    echo "CRITICAL: PERCEPT unhealthy"
    exit 2
fi

FPS=$(echo $HEALTH | jq -r '.fps')
if (( $(echo "$FPS < 5" | bc -l) )); then
    echo "WARNING: Low FPS ($FPS)"
    exit 1
fi

echo "OK: FPS=$FPS"
exit 0
```

### Alerting

Configure alerts for:

- Service down
- FPS below threshold
- CPU > 90%
- Temperature > 75°C
- Disk > 90%

---

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# /opt/percept/scripts/backup.sh

BACKUP_DIR=/var/backups/percept
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Stop service briefly
sudo systemctl stop percept

# Backup database
sqlite3 /var/lib/percept/percept.db ".backup '$BACKUP_DIR/percept_$DATE.db'"

# Backup embeddings
tar -czf $BACKUP_DIR/embeddings_$DATE.tar.gz /var/lib/percept/embeddings/

# Restart service
sudo systemctl start percept

# Keep only last 7 days
find $BACKUP_DIR -mtime +7 -delete
```

Add to cron:

```bash
# Run daily at 2 AM
0 2 * * * /opt/percept/scripts/backup.sh
```

### Recovery

```bash
# Restore database
sqlite3 /var/lib/percept/percept.db ".restore '/var/backups/percept/percept_YYYYMMDD.db'"

# Restore embeddings
tar -xzf /var/backups/percept/embeddings_YYYYMMDD.tar.gz -C /
```

### Configuration Backup

```bash
# Version control config
cd /opt/percept
git add config/
git commit -m "Config backup $(date +%Y%m%d)"
```

---

## Troubleshooting Deployment

### Service Won't Start

```bash
# Check logs
journalctl -u percept -n 100

# Test manually
cd /opt/percept
source venv/bin/activate
percept start --config config/production.yaml
```

### Permission Denied

```bash
# Fix ownership
sudo chown -R pi:pi /opt/percept /var/lib/percept /var/log/percept

# Check SELinux/AppArmor
sudo aa-status
```

### Out of Memory

```bash
# Check memory
free -h

# Reduce features
# Edit config: disable face_detection, pose_estimation
# Reduce gallery_size
```

### Temperature Throttling

```bash
# Monitor temperature
watch -n 1 vcgencmd measure_temp

# Check throttling
vcgencmd get_throttled
# 0x0 = OK
# Other values indicate throttling
```

---

## Updates

### Updating PERCEPT

```bash
cd /opt/percept
git pull origin main
source venv/bin/activate
pip install -e ".[production]"
sudo systemctl restart percept
```

### Database Migrations

```bash
percept migrate
```

### Rollback

```bash
git checkout <previous-commit>
pip install -e ".[production]"
sudo systemctl restart percept
```
