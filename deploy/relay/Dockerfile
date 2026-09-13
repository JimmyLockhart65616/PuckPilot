# The draft-view relay.
#
# Copies four files, installs nothing, and runs on the standard library alone -
# which is the point. The engine needs numpy, pandas and scipy; the relay never
# ranks anything, so an image carrying them would be ~400 MB of dependencies for
# a process that hands back JSON somebody else computed.
#
# Build from the REPO ROOT:
#   docker build -f deploy/relay/Dockerfile -t puckpilot-relay .
FROM python:3.12-slim

WORKDIR /app
COPY src/puckpilot/__init__.py      src/puckpilot/__init__.py
COPY src/puckpilot/web/__init__.py  src/puckpilot/web/__init__.py
COPY src/puckpilot/web/access.py    src/puckpilot/web/access.py
COPY src/puckpilot/web/page.py      src/puckpilot/web/page.py
COPY src/puckpilot/web/relay.py     src/puckpilot/web/relay.py

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Keys are never baked in - they arrive as container app secrets at runtime.
EXPOSE 8080
USER nobody
CMD ["python", "-m", "puckpilot.web.relay"]
