#!/usr/bin/env bash
# Ship the current checkout to an EXISTING relay, keeping its URL and its keys.
#
# deploy.sh stands everything up from nothing: a new registry name, a new
# environment, new keys - which on draft week means a new link to send and a
# console to restart. This only builds a new image into the registry the app
# already uses and points the app at it, then refuses to call it done until
# the live /healthz reports the build this checkout computes.
#
# A new revision starts with an empty board (state is in memory by design), so
# do not run this during a draft. Pushes resume on the console's next
# heartbeat, within ~10 seconds.
#
# Usage:  deploy/relay/update.sh            (from the repo root)
#         APP=puckpilot-draft-e2e deploy/relay/update.sh
set -euo pipefail

RG=${RG:-puckpilot-rg}
APP=${APP:-puckpilot-draft}
PY=${PY:-python}

# Same guard as deploy.sh: `az acr build` uploads the build context.
if ! head -n 20 .dockerignore 2>/dev/null | grep -qx '\*'; then
  echo "refusing to build: .dockerignore is missing or does not deny by default" >&2
  exit 1
fi

IMAGE=$(az containerapp show -n "$APP" -g "$RG" --query "properties.template.containers[0].image" -o tsv)
REGISTRY=${IMAGE%%/*}
ACR=${REGISTRY%%.*}
SHA=$(git rev-parse --short=12 HEAD)
if ! git diff --quiet HEAD -- src/puckpilot/web deploy/relay; then
  SHA="$SHA-dirty"
fi
TAG="git-$SHA"
EXPECT=$($PY -c "from puckpilot.web.relay import build_id; print(build_id())")

echo "==> $APP currently runs $IMAGE"
echo "==> building puckpilot-relay:$TAG in $ACR (expect build $EXPECT)"
az acr build --registry "$ACR" --image "puckpilot-relay:$TAG" \
  --build-arg "GIT_SHA=$SHA" --file deploy/relay/Dockerfile . -o none

echo "==> pointing $APP at it"
az containerapp update -n "$APP" -g "$RG" --image "$REGISTRY/puckpilot-relay:$TAG" -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "==> waiting for https://$FQDN/healthz to report $EXPECT"
for _ in $(seq 1 60); do
  GOT=$(curl -fsS "https://$FQDN/healthz" 2>/dev/null \
        | $PY -c "import json,sys; print(json.load(sys.stdin).get('build',''))" 2>/dev/null || true)
  if [ "$GOT" = "$EXPECT" ]; then
    echo "==> live: $APP serves build $EXPECT ($TAG)"
    echo "    rollback: az containerapp update -n $APP -g $RG --image $IMAGE"
    exit 0
  fi
  sleep 5
done
echo "timed out: /healthz reports '${GOT:-nothing}', expected $EXPECT" >&2
echo "rollback: az containerapp update -n $APP -g $RG --image $IMAGE" >&2
exit 1
