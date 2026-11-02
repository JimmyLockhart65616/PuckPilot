#!/usr/bin/env bash
# Create the in-season relay from nothing: resource group, state storage,
# environment, container app. Prints the URL and the per-manager keys.
#
# The image comes from GitHub's container registry, where
# .github/workflows/publish-season-image.yml publishes it - push first.
#
# Re-running is safe for the group/storage/environment, but the app is created
# here - use update.sh to redeploy an existing one and keep its URL and keys.
#
#   MANAGERS=jimmy,sam deploy/season/deploy.sh
set -euo pipefail

RG=${RG:-puckpilot-rg}
LOC=${LOC:-eastus2}
APP=${APP:-puckpilot-season}
ENVNAME=${ENVNAME:-puckpilot-cae}
STORE=${STORE:-puckpilot$RANDOM}
SHA=$(git log -1 --format=%H -- src/puckpilot/__init__.py src/puckpilot/web deploy/season \
  .github/workflows/publish-season-image.yml)
IMAGE=${IMAGE:-ghcr.io/jimmylockhart65616/puckpilot-season:${SHA:0:12}}
EXPIRY=${EXPIRY:-$(date -u -d '+2 years' +%Y-%m-%dT%H:%MZ)}
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

# Between runs the relay scales to zero, so the snapshot and the decisions on
# their way home live in one blob. The app's token hangs off a stored access
# policy, so it can be extended or revoked without minting a new one.
echo "==> state storage $STORE"
az storage account create -n "$STORE" -g "$RG" -l "$LOC" --sku Standard_LRS --kind StorageV2 \
  --min-tls-version TLS1_2 --allow-blob-public-access false -o none
az storage container create --account-name "$STORE" -n season --auth-mode key -o none
az storage container policy create --account-name "$STORE" -c season -n relay \
  --permissions rw --expiry "$EXPIRY" -o none
SAS=$(az storage container generate-sas --account-name "$STORE" -n season --policy-name relay -o tsv)
STATE_URL="https://$STORE.blob.core.windows.net/season/state.json?$SAS"

echo "==> container apps environment $ENVNAME"
# No Log Analytics workspace: console logs stream live (az containerapp logs show).
az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC" --logs-destination none \
  -o none 2>/dev/null || true

echo "==> container app $APP"
# At most one replica: a second would answer reads that never saw the push.
# At least none: the state is in the blob, so between runs the relay costs
# nothing, and the first request after a quiet spell waits a few seconds for it.
az containerapp create -n "$APP" -g "$RG" --environment "$ENVNAME" \
  --image "$IMAGE" \
  --target-port 8080 --ingress external \
  --cpu 0.25 --memory 0.5Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets "manager-keys=$KEYSPEC" "state-url=$STATE_URL" \
  --env-vars "PUCKPILOT_MANAGER_KEYS=secretref:manager-keys" \
    "PUCKPILOT_STATE_URL=secretref:state-url" \
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
echo "State:  $STORE/season/state.json; its token lapses $EXPIRY. Extend it with:"
echo "  az storage container policy update --account-name $STORE -c season -n relay --expiry <date>"
echo "Tear down with: az containerapp delete -n $APP -g $RG --yes"
