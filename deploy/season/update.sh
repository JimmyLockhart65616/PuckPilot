#!/usr/bin/env bash
# Redeploy the existing in-season relay, keeping its URL, its keys and its state.
#   deploy/season/update.sh
#
# .github/workflows/publish-season-image.yml builds the image when a push to
# in-season touches the relay. This deploys the image holding the last commit
# that did, waiting for its build if it is still running - so push first.
set -euo pipefail

RG=${RG:-puckpilot-rg}
APP=${APP:-puckpilot-season}
PY=${PY:-python}

SHA=$(git log -1 --format=%H -- src/puckpilot/__init__.py src/puckpilot/web deploy/season \
  .github/workflows/publish-season-image.yml)
git merge-base --is-ancestor "$SHA" "@{upstream}" 2>/dev/null \
  || { echo "push first: no image is built from ${SHA:0:12} until it is on GitHub" >&2; exit 1; }

# A push builds only its newest commit, so the relay of ${SHA} is in the image
# of the latest build whose commit descends from it and is in this checkout's
# history: every commit between the two leaves the relay's files as they are.
RUN= BUILT=
while read -r id head; do
  if git merge-base --is-ancestor "$SHA" "$head" 2>/dev/null \
    && git merge-base --is-ancestor "$head" HEAD 2>/dev/null; then
    RUN=$id BUILT=$head
    break
  fi
done < <(gh run list --workflow publish-season-image.yml --branch in-season --limit 30 \
  --json databaseId,headSha --jq '.[] | "\(.databaseId) \(.headSha)"')
[ -n "$RUN" ] || { echo "no image build holds ${SHA:0:12}" >&2; exit 1; }
IMAGE=${IMAGE:-ghcr.io/jimmylockhart65616/puckpilot-season:${BUILT:0:12}}
echo "==> waiting for the image build of ${BUILT:0:12} (run $RUN; relay as of ${SHA:0:12})"
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
