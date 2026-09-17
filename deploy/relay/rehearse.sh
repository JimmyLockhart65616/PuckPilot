#!/usr/bin/env bash
# Rehearse draft night against real Azure, without touching the real relay.
#
# Builds this checkout into the registry the production relay already uses,
# stands up a SEPARATE container app in the same environment with fresh keys,
# drives whole drafts through it with `ppilot draft e2e --expect-build`, and
# deletes the app afterwards. The production app, its URL, its keys and its
# in-memory board are never read or written.
#
# Keys are generated here and passed to the run through the environment only;
# they are never printed.
#
# Usage:  deploy/relay/rehearse.sh [ppilot draft e2e args...]
#         KEEP=1 deploy/relay/rehearse.sh --source sim --drafts 1   (leave the app up)
set -euo pipefail

RG=${RG:-puckpilot-rg}
PROD_APP=${PROD_APP:-puckpilot-draft}
APP=${APP:-puckpilot-draft-e2e}
PY=${PY:-python}
PPILOT=${PPILOT:-ppilot}

if [ "$APP" = "$PROD_APP" ]; then
  echo "refusing: the rehearsal app must not be the production app" >&2
  exit 1
fi
if ! head -n 20 .dockerignore 2>/dev/null | grep -qx '\*'; then
  echo "refusing to build: .dockerignore is missing or does not deny by default" >&2
  exit 1
fi

PROD_IMAGE=$(az containerapp show -n "$PROD_APP" -g "$RG" --query "properties.template.containers[0].image" -o tsv)
ENV_ID=$(az containerapp show -n "$PROD_APP" -g "$RG" --query "properties.environmentId" -o tsv)
# The name, not the id: Git Bash rewrites a leading "/subscriptions/..." into a
# Windows path before az sees it. The environment is in the same group anyway.
ENV_NAME=${ENV_ID##*/}
REGISTRY=${PROD_IMAGE%%/*}
ACR=${REGISTRY%%.*}
SHA=$(git rev-parse --short=12 HEAD)
git diff --quiet HEAD -- src/puckpilot/web deploy/relay || SHA="$SHA-dirty"
TAG="git-$SHA"
EXPECT=$($PY -c "from puckpilot.web.relay import build_id; print(build_id())")

cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then
    echo "==> deleting $APP"
    az containerapp delete -n "$APP" -g "$RG" --yes -o none || true
  else
    echo "==> KEEP=1: leaving $APP up (delete: az containerapp delete -n $APP -g $RG --yes)"
  fi
}

echo "==> building puckpilot-relay:$TAG in $ACR (expect build $EXPECT)"
az acr build --registry "$ACR" --image "puckpilot-relay:$TAG" \
  --build-arg "GIT_SHA=$SHA" --file deploy/relay/Dockerfile . -o none

OWNER_KEY=$($PY -c "import secrets;print(secrets.token_urlsafe(24))")
GUEST_KEY=$($PY -c "import secrets;print(secrets.token_urlsafe(24))")

trap cleanup EXIT
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  echo "==> $APP exists; updating image and keys"
  az containerapp secret set -n "$APP" -g "$RG" \
    --secrets "owner-key=$OWNER_KEY" "guest-key=$GUEST_KEY" -o none
  az containerapp update -n "$APP" -g "$RG" --image "$REGISTRY/puckpilot-relay:$TAG" \
    --revision-suffix "r$(date +%s)" -o none
else
  echo "==> creating $APP in the production environment"
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --image "$REGISTRY/puckpilot-relay:$TAG" \
    --registry-server "$REGISTRY" \
    --target-port 8080 --ingress external \
    --cpu 0.25 --memory 0.5Gi \
    --min-replicas 1 --max-replicas 1 \
    --secrets "owner-key=$OWNER_KEY" "guest-key=$GUEST_KEY" \
    --env-vars "PUCKPILOT_OWNER_KEY=secretref:owner-key" "PUCKPILOT_GUEST_KEY=secretref:guest-key" \
    -o none
fi

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "==> waiting for https://$FQDN/healthz to report $EXPECT"
GOT=""
for _ in $(seq 1 90); do
  GOT=$(curl -fsS "https://$FQDN/healthz" 2>/dev/null \
        | $PY -c "import json,sys; print(json.load(sys.stdin).get('build',''))" 2>/dev/null || true)
  [ "$GOT" = "$EXPECT" ] && break
  sleep 5
done
if [ "$GOT" != "$EXPECT" ]; then
  echo "rehearsal relay never came up on build $EXPECT (got '${GOT:-nothing}')" >&2
  exit 1
fi

echo "==> rehearsing against https://$FQDN"
PUCKPILOT_E2E_OWNER_KEY="$OWNER_KEY" PUCKPILOT_E2E_GUEST_KEY="$GUEST_KEY" \
  $PPILOT draft e2e --relay "https://$FQDN" --expect-build "$@"
