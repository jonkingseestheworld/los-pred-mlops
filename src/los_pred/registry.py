"""Model Registry promotion helper — practice the Candidate -> Staging ->
Production lifecycle for the LOS model.

MLflow 3.x note
===============
The classic *stages* (None / Staging / Production / Archived) are REMOVED in
MLflow 3. The modern, more flexible replacement is **aliases** + **tags**:

  - alias = a movable, named pointer to exactly one version — "which version
            currently plays this role". Reassigning an alias moves the pointer;
            code that loads by alias never needs to know the version number.
  - tag   = free-form key/value metadata on a version (validation status,
            who signed off, ticket link, ...).

We map the classic lifecycle onto aliases:

    Candidate   ->  @candidate   (freshly trained, not yet validated)
    Staging     ->  @staging     (passed offline checks, under evaluation)
    Production  ->  @champion     (the live model; community-standard name)

Because an alias points to exactly one version, promoting a new version to
@champion automatically supersedes the previous one — that's the "transition".

Load a role by alias (decoupled from version numbers — what serving/monitoring do):

    import mlflow.lightgbm
    model = mlflow.lightgbm.load_model("models:/los-pred-los@champion")

Practice run
------------
    uv run python -m los_pred.registry --show
    uv run python -m los_pred.registry --assign candidate --version 2
    uv run python -m los_pred.registry --assign staging   --version 2
    uv run python -m los_pred.registry --assign champion  --version 2
    uv run python -m los_pred.registry --show

Everything here is local (sqlite mlflow.db) and reversible — reassign or use
--remove to move pointers around freely while you learn.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from los_pred.train import REGISTERED_MODEL, TRACKING_URI

# Classic lifecycle stage -> MLflow 3 alias.
LIFECYCLE = {
    "candidate": "candidate",
    "staging": "staging",
    "production": "champion",  # community-standard name for the live model
}
ALIASES = list(LIFECYCLE.values())


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def load_champion(alias: str = "champion", name: str = REGISTERED_MODEL) -> dict:
    """Load the live model behind `@alias` from the registry.

    Returns {model (pyfunc), params (fitted transform state), log_target, version}
    — all from the SAME model version, so they can't drift apart. Shared by the
    serving app and the monitor so both operate on the exact model you serve.
    """
    client = _client()
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    params = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/transform_params.json")
    log_target = run.data.params.get("log_target", "False") == "True"
    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    return {"model": model, "params": params, "log_target": log_target,
            "version": mv.version}


def show(name: str = REGISTERED_MODEL) -> None:
    """Print every version with its status tag, and the current alias pointers."""
    client = _client()
    models = client.search_registered_models(filter_string=f"name='{name}'")
    if not models:
        print(f"no registered model named '{name}' — run train.py first")
        return
    model = models[0]
    print(f"MODEL: {model.name}")
    print(f"  aliases: {dict(model.aliases) or '(none)'}")
    for v in sorted(client.search_model_versions(f"name='{name}'"),
                    key=lambda v: int(v.version), reverse=True):
        status = v.tags.get("status", "-")
        print(f"  v{v.version:<3} status={status:<10} run={v.run_id[:8]}")


def assign(alias: str, version: int, name: str = REGISTERED_MODEL) -> None:
    """Point `alias` at `version` (the transition) and tag the version's status.

    `alias` accepts either a lifecycle name (candidate/staging/production) or a
    raw alias (candidate/staging/champion).
    """
    alias = LIFECYCLE.get(alias, alias)
    if alias not in ALIASES:
        raise SystemExit(f"unknown alias {alias!r}; choose from {ALIASES} "
                         f"(or lifecycle names {list(LIFECYCLE)})")
    client = _client()
    client.set_registered_model_alias(name, alias, str(version))
    # Mirror the alias onto the version as a human-readable status tag.
    client.set_model_version_tag(name, str(version), "status", alias)
    print(f"@{alias} -> {name} v{version}   (load: models:/{name}@{alias})")


def remove(alias: str, name: str = REGISTERED_MODEL) -> None:
    """Delete an alias pointer (does not delete the version)."""
    alias = LIFECYCLE.get(alias, alias)
    _client().delete_registered_model_alias(name, alias)
    print(f"removed @{alias} from {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice the model-registry promotion lifecycle.")
    parser.add_argument("--show", action="store_true", help="print versions and aliases")
    parser.add_argument("--assign", metavar="ALIAS",
                        help="alias/lifecycle stage to assign (candidate|staging|production|champion)")
    parser.add_argument("--version", type=int, help="model version to assign the alias to")
    parser.add_argument("--remove", metavar="ALIAS", help="remove an alias pointer")
    args = parser.parse_args()

    if args.remove:
        remove(args.remove)
    elif args.assign:
        if args.version is None:
            raise SystemExit("--assign requires --version N")
        assign(args.assign, args.version)
    if args.show or not (args.assign or args.remove):
        show()


if __name__ == "__main__":
    main()
