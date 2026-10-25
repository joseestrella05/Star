#!/usr/bin/env bash
# Demonstrates that the Star model server has no route to the internet.
# Run on the host after `docker compose -f deploy/docker-compose.yml up -d`.
set -uo pipefail
cd "$(dirname "$0")"
compose() { docker compose -f docker-compose.yml "$@"; }
in_app() { compose exec -T app python -c "$1"; }
fail=0
check() {  # check "<description>" <expected: pass|blocked> <python code>
  if in_app "$3" >/dev/null 2>&1; then got=pass; else got=blocked; fi
  if [ "$got" = "$2" ]; then echo "  OK    $1"; else echo "  FAIL  $1 (got: $got)"; fail=1; fi
}

echo "Star offline verification"
echo
echo "Network isolation of the model server (container 'app'):"
network=$(docker inspect "$(compose ps -q app)" --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}')
for n in $network; do
  echo "  network $n internal=$(docker network inspect "$n" --format '{{.Internal}}')"
done
check "DNS lookup of example.com fails" blocked \
  "import socket; socket.getaddrinfo('example.com', 443)"
check "HTTPS to 1.1.1.1 is blocked" blocked \
  "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)"
check "HTTPS to download.pytorch.org is blocked" blocked \
  "import urllib.request; urllib.request.urlopen('https://download.pytorch.org', timeout=5)"
check "Local API answers /health" pass \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"
check "Swagger /docs (CDN assets) is disabled" blocked \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=5)"

echo
echo "Model fingerprint served by this deployment:"
in_app "import json, urllib.request; h = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); print('  sha256', h['checkpoint_sha256'], '|', h['params'], 'params |', h['device'])"

echo
if [ "$fail" = 0 ]; then echo "RESULT: offline mode verified"; else echo "RESULT: FAILED"; fi
exit "$fail"
