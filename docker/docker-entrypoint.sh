#!/bin/bash
set -e

if [ "$#" -eq 0 ]; then
  set -- /app/start-web.sh
fi

detect_runtime_ids() {
  if [ -n "${APP_UID:-}" ] && [ -n "${APP_GID:-}" ]; then
    return
  fi

  for path in /app/data /app/frontend/backend/data /app/output; do
    if [ -e "$path" ]; then
      detected_uid="$(stat -c '%u' "$path" 2>/dev/null || true)"
      detected_gid="$(stat -c '%g' "$path" 2>/dev/null || true)"
      if [ -n "$detected_uid" ] && [ "$detected_uid" != "0" ]; then
        APP_UID="$detected_uid"
        APP_GID="${detected_gid:-$detected_uid}"
        return
      fi
    fi
  done

  APP_UID="${APP_UID:-1000}"
  APP_GID="${APP_GID:-1000}"
}

ensure_runtime_user() {
  if ! getent group "$APP_GID" >/dev/null; then
    groupadd -g "$APP_GID" appgroup
  fi

  APP_USER="$(getent passwd "$APP_UID" | cut -d: -f1 || true)"
  if [ -z "$APP_USER" ]; then
    APP_USER=appuser
    useradd -u "$APP_UID" -g "$APP_GID" -m -s /bin/bash "$APP_USER"
  fi
}

if [ "$(id -u)" -eq 0 ]; then
  detect_runtime_ids
  ensure_runtime_user

  mkdir -p /app/data /app/output /app/frontend/backend/data
  chown -R "$APP_UID:$APP_GID" /app/output /app/frontend/backend/data

  export APP_UID APP_GID
  exec runuser -u "$APP_USER" -- "$@"
fi

exec "$@"
