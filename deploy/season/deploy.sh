#!/usr/bin/env bash
# Create the in-season relay from nothing: resource group, registry,
# environment, container app. Prints the URL and the per-manager keys.
#
# Re-running is safe for the group/registry/environment, but the app is created
# here - use update.sh to redeploy an existing one and keep its URL and keys.
#
#   MANAGERS=jimmy,sam deploy/season/deploy.sh
set -euo pipefail

RG=${RG:-puckpilot-rg}
LOC=${LOC:-eastus2}
APP=${APP:-puckpilot-season}
ENVNAME=${ENVNAME:-puckpilot-cae}
ACR=${ACR:-puckpilot$RANDOM}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short=12 HEAD 2>/dev/null || date +%s)}
PY=${PY:-python}
MANAGERS=${MANAGERS:-jimmy}

# One key per manager. They are rivals in the same league, so a key admits you
# to your own view and nothing else.
KEYSPEC=""
declare -a SHOW=()
IFS=',' read -ra NAMES <<< "$MANAGERS"
for name in "${NAMES[@]}"; do
  key=$("$PY" -c "import secrets;print(secrets.token_urlsafe(16))")
  KEYSPEC="${KEYSPEC:+$KEYSPEC,}$name:$key"
  SHOW+=("$name $key")
done

echo "==> resource group $RG ($LOC)"
az group create -n "$RG" -l "$LOC" -o none

echo "==> container registry $ACR"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none

# The build context is the repo root, and `az acr build` uploads it. Without a
# deny-by-default .dockerignore this ships secrets/chrome-profile (a logged-in
# Yahoo session) and data/captures to a cloud registry.
if ! head -n 20 .dockerignore 2>/dev/null | grep -qx '\*'; then
  echo "refusing to build: .dockerignore is missing or does not deny by default" >&2
  exit 1
fi

echo "==> build image in ACR"
az acr build --registry "$ACR" --image "puckpilot-season:$IMAGE_TAG" \
  --file deploy/season/Dockerfile \
  --build-arg GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)" . -o none

echo "==> container apps environment $ENVNAME"
az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC" -o none 2>/dev/null || true

echo "==> container app $APP"
# One replica, never zero: the snapshot and any undelivered decisions live in
# memory, so a second instance would answer reads that never saw the push.
az containerapp create -n "$APP" -g "$RG" --environment "$ENVNAME" \
  --image "$ACR.azurecr.io/puckpilot-season:$IMAGE_TAG" \
  --registry-server "$ACR.azurecr.io" \
  --target-port 8080 --ingress external \
  --min-replicas 1 --max-replicas 1 \
  --secrets "manager-keys=$KEYSPEC" \
  --env-vars "PUCKPILOT_MANAGER_KEYS=secretref:manager-keys" \
  -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "URL     https://$FQDN"
echo "health  https://$FQDN/healthz"
for row in "${SHOW[@]}"; do
  set -- $row
  echo "$1      https://$FQDN/?k=$2"
done
echo
echo "Registry: ACR=$ACR (needed by update.sh)"
echo "Tear down with: az containerapp delete -n $APP -g $RG --yes"
