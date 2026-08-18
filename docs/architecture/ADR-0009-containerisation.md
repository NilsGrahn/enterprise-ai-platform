# ADR-0009 — Containerisation and compose orchestration

## Status
Accepted

## Context
The platform ran only on a machine that had been configured over several weeks:
WSL, Postgres, a Python virtual environment, a populated database, a trained
model, and two manually started services. Reproducing that elsewhere required a
long sequence of manual steps, any of which could differ and produce a different
result.

The services also have a genuine dependency order — data must be loaded before a
model can be trained, and a model must exist before the API can serve — which was
previously enforced only by the operator remembering it.

## Decision
1. **A shared base image** (`Dockerfile.base`) installs the operating system
   packages and all Python dependencies once. Three thin service images build on
   it, adding only code.

2. **Each service image copies only the packages that service imports.** The
   dashboard image contains no `ml-service/`, so the ADR-0007 rule that the
   dashboard must not load a model is enforced by the image contents rather than
   by discipline — an added import would fail at startup.

3. **Dependencies are installed before source is copied**, so Docker's layer
   cache means editing a `.py` file does not re-run `pip install`.

4. **`libgomp1` is installed explicitly.** XGBoost links against the OpenMP
   runtime and slim Python images omit it.

5. **Artifacts and raw data are bind-mounted, not baked in.** Retraining does not
   require rebuilding an image, and the model written by the trainer container is
   visible on the host. The API mounts `artifacts` read-only, and the ETL mounts
   `data` read-only, so the separation between writing and reading artifacts is
   enforced by the mount.

6. **The database uses a named volume**, so it survives `docker compose down`.
   `down -v` is the deliberate full reset.

7. **Batch jobs and long-running services are distinguished.** `etl`, `trainer`
   and `reference` have `restart: "no"` and exit 0. `api` and `dashboard` have
   `restart: unless-stopped`.

8. **`service_completed_successfully` and `service_healthy` encode the real
   ordering**: postgres healthy → etl exits 0 → trainer exits 0 → reference
   exits 0 → api healthy → dashboard. One `docker compose up` on an empty
   database produces a serving platform.

9. **A `reference` service was added** beyond the original design, so a clean
   deployment also produces `reference_profile.json`. Without it the Phase 7
   drift check fails on a fresh environment.

10. **Every connection detail is an environment variable**, and services address
    each other by compose service name. Because this was already true from Phase
    0, containerisation required no application code changes at all.

11. **Postgres is published on host port 5433**, avoiding a clash with the WSL
    Postgres installed in Phase 0. Containers still use 5432 internally.

12. **`CMD` uses exec form**, so the process is PID 1 and receives termination
    signals directly — which is what allows the FastAPI lifespan shutdown hook to
    run.

## Alternatives considered
- **One image containing everything.** Simpler, but larger, and it would destroy
  the architectural guarantee that the dashboard cannot import the model layer.
- **Baking artifacts into the API image.** Removes a mount, but makes every
  retrain an image rebuild and couples model versioning to image versioning.
- **`depends_on` without conditions.** Waits only for containers to exist, so the
  ETL would race Postgres's initialisation.
- **A shell script running the services in order.** Works locally, but does not
  produce reproducible environments and does not survive being handed to someone
  else.
- **Kubernetes.** The right answer at scale and enormous overkill for five
  services on one machine.

## Consequences
- The raw dataset is gitignored, so `docker compose up` on a truly clean clone
  fails at the ETL step until the CSV is placed in `data/raw/`. This is a
  documented prerequisite in the README rather than a defect.
- The compose file lives in `infrastructure/` while `.env` is at the repo root,
  so compose needs `--env-file ../.env`. Moving the compose file to the root
  would remove the friction at the cost of a less tidy layout.
- `pip install -e .` in each service image reads a `pyproject.toml` that lists
  all six source directories, including ones the image did not copy. Setuptools
  currently skips missing directories, so this works, but it is fragile. Proper
  per-service packaging metadata would be the robust fix and is deferred.
- The container runs Python 3.11 while local development uses 3.14, so the
  container is not an exact replica of the development environment.
- `requirements.txt` is unpinned, so an image built today and one built in six
  months may contain different package versions. Generating a lock file would fix
  this and is deferred.
- Containers run as root. Acceptable locally; a production image should create
  and switch to an unprivileged user.
- There is still no authentication on the API, now exposed on a published port.
  Recorded in the limitations section of `architecture.md`.

## Date
2026-08-15