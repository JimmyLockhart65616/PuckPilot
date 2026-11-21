# The in-season view relay.
#
# Same principle as the draft relay: copies a handful of files, installs
# nothing, and runs on the standard library alone. The engine needs numpy,
# pandas and scipy; this process ranks nothing - it holds the last snapshot a
# local job pushed and carries decisions back - so an image carrying them would
# be ~400 MB of dependencies for something that hands back JSON.
#
# Build from the REPO ROOT:
#   docker build -f deploy/season/Dockerfile -t puckpilot-season .
FROM python:3.12-slim

WORKDIR /app
COPY src/puckpilot/__init__.py          src/puckpilot/__init__.py
COPY src/puckpilot/web/__init__.py      src/puckpilot/web/__init__.py
COPY src/puckpilot/web/season_page.py   src/puckpilot/web/season_page.py
COPY src/puckpilot/web/season_relay.py  src/puckpilot/web/season_relay.py
COPY src/puckpilot/web/wire.py          src/puckpilot/web/wire.py

# Stamped at build so /healthz can name the commit it was built from.
ARG GIT_SHA=""
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PUCKPILOT_GIT_SHA=${GIT_SHA} \
    PORT=8080

# Keys are never baked in - they arrive as container app secrets at runtime.
EXPOSE 8080
USER nobody
CMD ["python", "-m", "puckpilot.web.season_relay"]
