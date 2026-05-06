#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

set -eu

required_env="NC_DOMAIN HPB_DOMAIN RECORDING_SECRET INTERNAL_SECRET"
for variable in $required_env; do
	if [ -z "${!variable:-}" ]; then
		echo "Missing required environment variable: $variable" >&2
		exit 1
	fi
done

nc_url="${NC_DOMAIN}"
case "$nc_url" in
	http://*|https://*) ;;
	*) nc_url="https://${nc_url}" ;;
esac

hpb_url="${HPB_DOMAIN}"
case "$hpb_url" in
	http://*|https://*|ws://*|wss://*) ;;
	*) hpb_url="https://${hpb_url}" ;;
esac

log_level="${LOG_LEVEL:-20}"
listen="${LISTEN:-0.0.0.0:8000}"
recording_directory="${RECORDING_DIRECTORY:-/tmp}"
recording_browser="${RECORDING_BROWSER:-firefox}"

cat > /etc/nextcloud-talk-recording/server.conf <<EOF
[logs]
level = ${log_level}

[http]
listen = ${listen}

[backend]
backends = nextcloud
directory = ${recording_directory}

[nextcloud]
url = ${nc_url}
secret = ${RECORDING_SECRET}

[signaling]
signalings = hpb

[hpb]
url = ${hpb_url}
internalsecret = ${INTERNAL_SECRET}

[recording]
browser = ${recording_browser}
EOF

chown recording:recording /etc/nextcloud-talk-recording/server.conf

exec runuser -u recording -- python3 -m nextcloud.talk.recording --config /etc/nextcloud-talk-recording/server.conf
