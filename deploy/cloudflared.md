# Cloudflare Quick Tunnel

RoadGuard exposes its FastAPI service to the public internet through a
Cloudflare "try" quick tunnel. The tunnel URL is ephemeral (changes on
every restart) and the webapp picks it up from
`webapp/assets/config.js`.

No Cloudflare account, no DNS record, and no paid plan is required.

## Install `cloudflared` on Debian 13

```bash
sudo curl -L -o /usr/local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
sudo chmod +x /usr/local/bin/cloudflared
cloudflared --version
```

## Run a quick tunnel pointing at the FastAPI service

```bash
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate --metrics 127.0.0.1:9000
```

The output includes a line like:

```
2026-10-04T15:12:34Z INF +--------------------------------------------------+
2026-10-04T15:12:34Z INF |  Your quick Tunnel has been created!             |
2026-10-04T15:12:34Z INF |  https://<random-name>.trycloudflare.com         |
2026-10-04T15:12:34Z INF +--------------------------------------------------+
```

That URL is the public origin for the webapp.

## Run it as a systemd service

Create `/etc/systemd/system/cloudflared.service`:

```ini
[Unit]
Description=Cloudflare quick tunnel for RoadGuard
After=network-online.target roadguard.service
Wants=network-online.target

[Service]
Type=simple
User=roadguard
Group=roadguard
ExecStart=/usr/local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate --metrics 127.0.0.1:9000
Restart=on-failure
RestartSec=10
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared.service
sudo systemctl start cloudflared.service
journalctl -u cloudflared.service --since "-5 min" | grep trycloudflare
```

## Rotating the URL

The quick tunnel URL changes on every restart. The webapp reads it
from `webapp/tunnel-url.json` under GitHub Pages. On every restart,
the VM should:

1. Parse the new tunnel URL from `cloudflared` output.
2. Commit `{"origin": "https://<new>.trycloudflare.com"}` to
   `webapp/tunnel-url.json` on the `main` branch, which triggers the
   GitHub Pages workflow.

The commit script lives at `deploy/update-tunnel-url.sh` (added in a
later sprint) and is invoked by a `cloudflared-url-watcher.service`.

## Health check

```bash
curl -s "https://<the-tunnel>.trycloudflare.com/health" | jq .
```

Returns `{"status": "ok", ...}` when the service is up and healthy.
