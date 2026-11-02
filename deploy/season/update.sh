#!/usr/bin/env bash
# Redeploy the existing in-season relay, keeping its URL, its keys and its state.
#   deploy/season/update.sh
#
# .github/workflows/publish-season-image.yml builds the image when a push to
# in-season touches the relay. This deploys the image of the last commit that
# did, waiting for its build if it is still running - so push first.
set -euo pipefail

RG=${RG:-puckpilot-rg}
APP=${APP:-puckpilot-season}
PY=${PY:-python}

SHA=$(git log -1 --format=%H -- src/puckpilot/__init__.py src/puckpilot/web deploy/season \
  .github/workflows/publish-season-image.yml)
git merge-base --is-ancestor "$SHA" "@{upstream}" 2>/dev/null \
  || { echo "push first: no image is built from ${SHA:0:12} until it is on GitHub" >&2; exit 1; }
IMAGE=${IMAGE:-ghcr.io/jimmylockhart65616/puckpilot-season:${SHA:0:12}}

RUN=$(gh run list --workflow publish-season-image.yml --commit "$SHA" --limit 1 \
  --json databaseId --jq '.[0].databaseId // empty')
[ -n "$RUN" ] || { echo "no image build ran for ${SHA:0:12}" >&2; exit 1; }
echo "==> waiting for the image build of ${SHA:0:12} (run $RUN)"
gh run watch "$RUN" --exit-status > /dev/null

BUILD=$("$PY" -c "import sys;sys.path.insert(0,'src');from puckpilot.web.season_relay import build_id;print(build_id())")
echo "==> local build id $BUILD"

PREV=$(az containerapp show -n "$APP" -g "$RG" --query properties.template.containers[0].image -o tsv)
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "deployed. check it serves what this checkout computes (the first call may wait"
echo "a few seconds for the relay to start):"
echo "  curl -s https://$FQDN/healthz   # build should be $BUILD"
echo "rollback: az containerapp update -n $APP -g $RG --image $PREV"
