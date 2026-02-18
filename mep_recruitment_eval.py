# mep_recruitment_eval.py
"""
Evaluation utilities for mep_recruitment.py, analogous to location_finding_eval.py
from the Deep Adaptive Design (DAD) repo.

Computes Monte Carlo estimates of:
- a LOWER bound on EIG via Prior Contrastive Estimation (PCE), and
- an UPPER bound on EIG via Nested Monte Carlo (NMC),

for a trained design network saved as an MLflow artifact.
"""

import os
import math
import argparse

import pandas as pd

import torch
import pyro

import mlflow
import mlflow.pytorch

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from contrastive.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation

# Import the model definition so we can override T when desired.
from mep_recruitment import MEPModel  # noqa: F401


def _load_mep_model(experiment_id: str, run_id: str, device: str):
    model_location = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    mep_model = mlflow.pytorch.load_model(model_location, map_location=device)
    # Ensure design_net is on the right device
    try:
        mep_model.to(device)
    except Exception:
        pass
    mep_model.eval()
    return mep_model


def evaluate_run(
    experiment_id: str,
    run_id: str,
    num_experiments_to_perform,
    num_inner_samples: int,
    device: str,
    n_rollout: int,
    seed: int = -1,
):
    """
    Evaluate a single MLflow run, returning a DataFrame with mean/se
    for lower/upper bounds across desired horizons (T).

    We use the same convention as location_finding_eval.py:
      - set factor=16
      - run n_rollout//factor repetitions
      - each repetition uses `factor` outer samples inside the estimator
      => total outer samples = (n_rollout//factor) * factor
    """
    pyro.clear_param_store()
    seed = auto_seed(seed)

    factor = 16
    n_rollout_eff = max(1, n_rollout // factor)

    EIGs_mean = pd.DataFrame(columns=["lower", "upper"])
    EIGs_se = pd.DataFrame(columns=["lower", "upper"])

    mep_model = _load_mep_model(experiment_id, run_id, device=device)

    for t_exp in num_experiments_to_perform:
        if t_exp is not None:
            mep_model.T = int(t_exp)
        t_used = int(mep_model.T)

        # Upper bound: Nested Monte Carlo
        pce_loss_upper = NestedMonteCarloEstimation(factor, num_inner_samples)
        # Lower bound: PCE
        pce_loss_lower = PriorContrastiveEstimation(factor, num_inner_samples)

        auto_seed(seed)
        with torch.no_grad():
            EIG_proxy_upper = torch.tensor(
                [-pce_loss_upper.loss(mep_model.model) for _ in range(n_rollout_eff)],
                device="cpu",
            )

        auto_seed(seed)
        with torch.no_grad():
            EIG_proxy_lower = torch.tensor(
                [-pce_loss_lower.loss(mep_model.model) for _ in range(n_rollout_eff)],
                device="cpu",
            )

        EIGs_mean.loc[t_used, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_mean.loc[t_used, "upper"] = EIG_proxy_upper.mean().item()
        EIGs_se.loc[t_used, "lower"] = EIG_proxy_lower.std(unbiased=True).item() / math.sqrt(
            n_rollout_eff
        )
        EIGs_se.loc[t_used, "upper"] = EIG_proxy_upper.std(unbiased=True).item() / math.sqrt(
            n_rollout_eff
        )

    EIGs_mean["stat"] = "mean"
    EIGs_se["stat"] = "se"
    res = pd.concat([EIGs_mean, EIGs_se])

    print("\n")
    print(res)

    os.makedirs("mlflow_outputs", exist_ok=True)
    out_csv = "mlflow_outputs/eval.csv"
    res.to_csv(out_csv)

    # Log to MLflow under the same run
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        mlflow.log_param("n_rollouts", n_rollout_eff * factor)
        mlflow.log_param("eval_seed", seed)
        mlflow.log_param("eval_num_inner_samples", num_inner_samples)
        mlflow.log_artifact(out_csv, artifact_path="evaluation")

        if len(num_experiments_to_perform) == 1:
            t0 = int(mep_model.T)
            mlflow.log_metric("eval_EIG_lower", float(EIGs_mean.loc[t0, "lower"]))
            mlflow.log_metric("eval_EIG_upper", float(EIGs_mean.loc[t0, "upper"]))

    return res


def evaluate_experiment(
    experiment_id: str,
    run_id: str = None,
    num_experiments_to_perform=[None],
    num_inner_samples: int = int(2e5),
    device: str = "cpu",
    n_rollout: int = 2048,
    seed: int = -1,
):
    """
    Evaluate one run (if run_id provided) or evaluate all FINISHED runs in an MLflow experiment
    that have not yet been evaluated (i.e., no eval_seed logged).
    """
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string="")

    if run_id is not None:
        meta = [m for m in meta if m.info.run_id == run_id]
        if len(meta) == 0:
            raise ValueError(f"run_id={run_id} not found in experiment_id={experiment_id}")
    else:
        meta = [m for m in meta if "eval_seed" not in m.data.params.keys()]

    run_ids = [m.info.run_id for m in meta]
    if len(run_ids) == 0:
        print("No runs to evaluate (either none found, or all already evaluated).")
        return None

    results = {}
    for rid in run_ids:
        print(f"\nEvaluating experiment_id={experiment_id}, run_id={rid}")
        results[rid] = evaluate_run(
            experiment_id=experiment_id,
            run_id=rid,
            num_experiments_to_perform=num_experiments_to_perform,
            num_inner_samples=num_inner_samples,
            device=device,
            n_rollout=n_rollout,
            seed=seed,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design evaluation: MEP recruitment curve model."
    )
    parser.add_argument("--experiment-id", required=True, type=str)
    parser.add_argument("--run-id", default=None, type=str, help="Evaluate a specific MLflow run id.")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--n-rollout", default=2048, type=int)
    parser.add_argument("--num-inner-samples", default=int(2e5), type=int)
    parser.add_argument(
        "--num-experiments-to-perform",
        nargs="+",
        default=[None],
        help="List of horizons T to evaluate. Use empty string for 'use trained T'.",
    )

    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x not in [None, "None", ""] else None for x in args.num_experiments_to_perform
    ]

    evaluate_experiment(
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        n_rollout=args.n_rollout,
        num_inner_samples=args.num_inner_samples,
        num_experiments_to_perform=args.num_experiments_to_perform,
        device=args.device,
        seed=args.seed,
    )
