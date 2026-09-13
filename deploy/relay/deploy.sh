#!/usr/bin/env bash
# Stand up the draft-view relay on its own Azure infrastructure.
#
# Deliberately its own resource group: this is a personal tool with a different
# lifecycle from anything else in the subscription, and `az group delete -n
# puckpilot-rg` is then the entire teardown.
#
# Usage:  deploy/relay/deploy.sh            (from the repo root)
set -euo pipefail

RG=${RG:-puckpilot-rg}
LOC=${LOC:-eastus2}
ENVNAME=${ENVNAME:-puckpilot-cae}
APP=${APP:-puckpilot-draft}
ACR=${ACR:-puckpilot$RANDOM}
IMAGE_TAG=${IMAGE_TAG:-v1}

echo "==> keys"
OWNER_KEY=${PUCKPILOT_OWNER_KEY:-$(python -c "import secrets;print(secrets.token_urlsafe(16))")}
GUEST_KEY=${PUCKPILOT_GUEST_KEY:-$(python -c "import secrets;print(secrets.token_urlsafe(16))")}

echo "==> resource group $RG in $LOC"
az group create -n "$RG" -l "$LOC" -o none

echo "==> container registry $ACR"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none

# The context is the repo root, and `az acr build` uploads it to a cloud
# registry. `.dockerignore` at the root is deny-by-default for exactly that
# reason - without it this line ships secrets/chrome-profile (a logged-in Yahoo
# session) and data/captures (>1 GB of draft-room recordings) to ACR.
if ! head -n 20 .dockerignore 2>/dev/null | grep -qx '\*'; then
  echo "refusing to build: .dockerignore is missing or does not deny by default" >&2
  exit 1
fi

echo "==> build image (in ACR - no local docker needed)"
az acr build --registry "$ACR" --image "puckpilot-relay:$IMAGE_TAG" \
  --file deploy/relay/Dockerfile . -o none

echo "==> container apps environment $ENVNAME (this is the slow step, ~5-10 min)"
az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC" -o none

# min-replicas 1: the relay holds the board in memory, so scaling to zero
# mid-draft discards it. max-replicas 1: with two replicas the console's pushes
# land on one instance while the guest reads the other, and the board appears to
# flicker backwards. Both matter more than the pennies they cost.
echo "==> container app $APP"
az containerapp create -n "$APP" -g "$RG" --environment "$ENVNAME" \
  --image "$ACR.azurecr.io/puckpilot-relay:$IMAGE_TAG" \
  --registry-server "$ACR.azurecr.io" \
  --target-port 8080 --ingress external \
  --min-replicas 1 --max-replicas 1 \
  --secrets "owner-key=$OWNER_KEY" "guest-key=$GUEST_KEY" \
  --env-vars "PUCKPILOT_OWNER_KEY=secretref:owner-key" \
             "PUCKPILOT_GUEST_KEY=secretref:guest-key" \
  -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)

cat <<SUMMARY

  Relay:      https://$FQDN
  Health:     https://$FQDN/healthz

  Run the console with:
    export PUCKPILOT_OWNER_KEY=$OWNER_KEY
    ppilot draft live --seat <yours> --yahoo <league-key> --web \
        --seats <yours>,<his> --publish https://$FQDN

  Send him:   https://$FQDN/?seat=<his>&k=$GUEST_KEY

  Afterwards: az containerapp update -n $APP -g $RG --min-replicas 0
  Teardown:   az group delete -n $RG --yes

SUMMARY
