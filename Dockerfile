ARG PRESSLOUPE_DEPS_IMAGE=python:3.11-slim
FROM ${PRESSLOUPE_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/pressloupe
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 PRESSLOUPE_DATA=/var/lib/pressloupe
WORKDIR /opt/pressloupe
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY pressloupe ./pressloupe
COPY static ./static
RUN useradd -U -u 11513 pressloupe && install -d -o pressloupe -g pressloupe /var/lib/pressloupe
USER pressloupe
EXPOSE 8000
CMD ["python", "-m", "pressloupe"]
