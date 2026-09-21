#!/usr/bin/env bash
# Redeploy the existing in-season relay, keeping its URL and its keys.
#   ACR=<registry> deploy/season/update.sh
set -euo pipefail

RG=${RG:-puckpilot-rg}
APP=${APP:-puckpilot-season}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short=12 HEAD 2>/dev/null || date +%s)}
PY=${PY:-python}

ACR=${ACR:-$(az acr list -g "$RG" --query "[0].name" -o tsv)}
[ -n "$ACR" ] || { echo "no registry in $RG; set ACR=" >&2; exit 1; }

if ! head -n 20 .dockerignore 2>/dev/null | grep -qx '\*'; then
  echo "refusing to build: .dockerignore is missing or does not deny by default" >&2
  exit 1
fi

BUILD=$("$PY" -c "import sys;sys.path.insert(0,'src');from puckpilot.web.season_relay import build_id;print(build_id())")
echo "==> local build id $BUILD"

az acr build --registry "$ACR" --image "puckpilot-season:$IMAGE_TAG" \
  --file deploy/season/Dockerfile \
  --build-arg GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)" . -o none

PREV=$(az containerapp show -n "$APP" -g "$RG" --query properties.template.containers[0].image -o tsv)
az containerapp update -n "$APP" -g "$RG" \
  --image "$ACR.azurecr.io/puckpilot-season:$IMAGE_TAG" -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "deployed. check it serves what this checkout computes:"
echo "  curl -s https://$FQDN/healthz   # build should be $BUILD"
echo "rollback: az containerapp update -n $APP -g $RG --image $PREV"
