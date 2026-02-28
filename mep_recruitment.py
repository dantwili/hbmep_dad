# mep_recruitment.py
"""
MEP Recruitment Curve Estimation using Deep Adaptive Design
Following the DAD framework structure from location_finding.py
"""

import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.distributions import transforms
from pyro.infer.util import torch_item

from tqdm import trange

import mlflow
import mlflow.pytorch

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
    RandomDesignBaseline,
)

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Network Architecture
# ============================================================================

class EncoderNetwork(nn.Module):
    """Encoder network for (intensity, MEP size) pairs"""

    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        # design_dim is (batch, 1) for intensity
        # observation_dim is (batch, 1) for MEP size
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.observation_dim_flat = observation_dim[0] * observation_dim[1]
        input_dim = self.design_dim_flat + self.observation_dim_flat

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.relu = nn.ReLU()

        # self.linear1 = nn.Linear(input_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        # self.softplus = nn.Softplus()

    def forward(self, xi, y, **kwargs):
        """
        Args:
            xi: design (intensity) [batch_size, n, 1]
            y: observation (MEP size) [batch_size, n]
        """
        xi = xi.squeeze(-1)  # [100, 1, 1] -> [100, 1]
        inputs = torch.cat([xi, y], dim=-1)

        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.output_layer(x)

        # x = self.linear1(inputs)
        # x = self.softplus(x)
        # x = self.linear2(x)
        # x = self.softplus(x)
        # x = self.output_layer(x)

        return x


class EmitterNetwork(nn.Module):
    """Emitter network that outputs next stimulation intensity"""

    def __init__(self, encoding_dim, design_dim, intensity_min=0.0, intensity_max=100.0):
        super().__init__()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        
        self.linear1 = nn.Linear(encoding_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, self.design_dim_flat)
        self.softplus = nn.Softplus()

    def forward(self, r):
        """
        Args:
            r: pooled representation [batch_size, encoding_dim]
        """
        x = self.linear1(r)
        x = self.softplus(x)
        x = self.linear2(x)
        x = self.softplus(x)
        xi_raw = self.output_layer(x)
        
        # Map to valid intensity range using sigmoid
        xi_normalized = torch.sigmoid(xi_raw)
        xi = self.intensity_min + (self.intensity_max - self.intensity_min) * xi_normalized
        
        return xi.reshape(xi.shape[:-1] + self.design_dim)


# ============================================================================
# MEP Model
# ============================================================================

def rectified_logistic(xi, a, b, L, ell, H):
    """
    Rectified-logistic recruitment curve function from hbMEP paper.
    
    Args:
        xi: stimulation intensity
        a: threshold
        b: growth rate  
        L: offset (baseline)
        ell: inflection location parameter
        H: saturation height
    """
    z = b * (xi - a)
    logistic_part = (H + ell) * torch.sigmoid(z - torch.log(H / ell))
    rectified = torch.relu(logistic_part - ell)
    return L + rectified


class MEPModel(nn.Module):
    """MEP recruitment curve experimental design model"""

    def __init__(
        self,
        design_net,
        a_loc,          # threshold prior mean
        a_scale,        # threshold prior scale
        b_scale,        # growth rate prior 
        L_scale,        # offset prior 
        ell_scale,       # inflection location prior 
        H_scale,         # saturation prior 
        c1_scale,        # noise parameter 1 
        c2_scale,        # noise parameter 2 
        intensity_min=0.0,
        intensity_max=100.0,
        T=10,
    ):
        super().__init__()
        self.design_net = design_net
        
        # Prior hyperparameters
        self.a_loc = torch.tensor(a_loc, device=device)
        self.a_scale = torch.tensor(a_scale, device=device)
        self.b_scale = torch.tensor(b_scale, device=device)
        self.L_scale = torch.tensor(L_scale, device=device)
        self.ell_scale = torch.tensor(ell_scale, device=device)
        self.H_scale = torch.tensor(H_scale, device=device)
        self.c1_scale = torch.tensor(c1_scale, device=device)
        self.c2_scale = torch.tensor(c2_scale, device=device)
        
        # Intensity range
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        
        # Experiment parameters
        self.n = 1  # batch size
        self.T = T  # number of experiments

    def model(self):
        """Generative model following DAD framework"""
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables (recruitment curve parameters)
        ########################################################################

        eps = 1e-6

        a_base = latent_sample("a_base", dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)))
        a = self.a_loc + self.a_scale * a_base
        a = torch.clamp(a, self.intensity_min, self.intensity_max)

        b = latent_sample("b", dist.HalfNormal(self.b_scale)).clamp(min=eps)
        ell = latent_sample("ell", dist.HalfNormal(self.ell_scale)).clamp(min=eps)
        H = latent_sample("H", dist.HalfNormal(self.H_scale)).clamp(min=eps)
        L = latent_sample("L", dist.HalfNormal(self.L_scale)).clamp(min=eps)
        c1 = latent_sample("c1", dist.HalfNormal(self.c1_scale)).clamp(min=eps)
        c2 = latent_sample("c2", dist.HalfNormal(self.c2_scale)).clamp(min=eps)

        # b = self.b_scale
        # ell = self.ell_scale
        # L = self.L_scale
        # H = self.H_scale
        # c1 = self.c1_scale
        # c2 = self.c2_scale

        # Reshape parameters to match xi dimensions [batch, n, 1]
        a = a.unsqueeze(-1).unsqueeze(-1)      # [100] -> [100, 1, 1]
        b = b.unsqueeze(-1).unsqueeze(-1)
        L = L.unsqueeze(-1).unsqueeze(-1)
        ell = ell.unsqueeze(-1).unsqueeze(-1)
        H = H.unsqueeze(-1).unsqueeze(-1)
        c1 = c1.unsqueeze(-1).unsqueeze(-1)
        c2 = c2.unsqueeze(-1).unsqueeze(-1)
        
        # Store all parameters together
        theta = dict(a=a, b=b, L=L, ell=ell, H=H, c1=c1, c2=c2)
        
        y_outcomes = []
        xi_designs = []

        ########################################################################
        # T-step experiment
        ########################################################################
        for t in range(self.T):
            ####################################################################
            # Get design (stimulation intensity)
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}",
                self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )

            ####################################################################
            # Compute expected MEP size using recruitment curve
            ####################################################################
            mu = rectified_logistic(xi, a, b, L, ell, H)
            
            ####################################################################
            # Gamma observation model (shape-rate parametrization)
            ####################################################################
            # Rate parameter depends on mean
            beta = 1/c1 + 1/(c2 * mu)
            concentration = (mu * beta).squeeze(-1)
            rate = beta.squeeze(-1)

            # Ensure positive parameters and squeeze to [100, 1]
            #concentration = torch.clamp(concentration, min=1e-4, max=100.0).squeeze(-1)
            #rate = torch.clamp(rate, min=1e-4, max=100.0).squeeze(-1)

            ####################################################################
            # Sample observation (MEP size)
            ####################################################################
            y = observation_sample(
                f"y{t + 1}",
                dist.Gamma(concentration, rate).to_event(1)
            )

            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def forward(self, theta=None):
        """Run the policy with optional conditioning on theta"""
        self.design_net.eval()
        
        if theta is not None:
            model = pyro.condition(self.model, data=theta)
        else:
            model = self.model
            
        designs = []
        observations = []

        with torch.no_grad():
            trace = pyro.poutine.trace(model).get_trace()
            for t in range(self.T):
                xi = trace.nodes[f"xi{t + 1}"]["value"]
                designs.append(xi)

                y = trace.nodes[f"y{t + 1}"]["value"]
                observations.append(y)
                
        return torch.cat(designs).unsqueeze(1), torch.cat(observations).unsqueeze(1)

    def eval(self, n_trace=3, theta=None, verbose=True):
        """Run the policy, print output and return in a pandas DataFrame"""
        self.design_net.eval()
        
        if theta is not None:
            model = pyro.condition(self.model, data=theta)
        else:
            model = self.model

        output = []
        true_params = []
        
        with torch.no_grad():
            for i in range(n_trace):
                print(f"\nExample run {i + 1}")
                trace = pyro.poutine.trace(model).get_trace()
                
                # Extract true parameters
                a_base = trace.nodes["a_base"]["value"].cpu().item()
                a = (self.a_loc + self.a_scale * a_base).item()
                a = np.clip(a, self.intensity_min, self.intensity_max)
                
                true_theta = {
                    'a': a,
                    # 'b': trace.nodes["b"]["value"].cpu().item(),
                    # 'L': trace.nodes["L"]["value"].cpu().item(),
                    # 'ell': trace.nodes["ell"]["value"].cpu().item(),
                    # 'H': trace.nodes["H"]["value"].cpu().item(),
                    # 'c1': trace.nodes["c1"]["value"].cpu().item(),
                    # 'c2': trace.nodes["c2"]["value"].cpu().item(),
                }
                
                if verbose:
                    print(f"*True Parameters:*")
                    print(f"  Threshold (a): {true_theta['a']:.2f}")
                    # print(f"  Growth rate (b): {true_theta['b']:.3f}")
                    # print(f"  Offset (L): {true_theta['L']:.3f}")
                    # print(f"  Saturation (L+H): {(true_theta['L'] + true_theta['H']):.3f}")
                
                run_xis = []
                run_ys = []
                
                # Print designs and observations
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].cpu().item()
                    run_xis.append(xi)
                    
                    y = trace.nodes[f"y{t + 1}"]["value"].cpu().item()
                    run_ys.append(y)
                    
                    if verbose:
                        print(f"xi{t + 1}: {xi:.2f} % MSO")
                        print(f" y{t + 1}: {y:.4f} mV")

                run_df = pd.DataFrame({
                    'intensity': run_xis,
                    'mep_size': run_ys,
                    'order': list(range(1, self.T + 1)),
                    'run_id': i + 1
                })
                output.append(run_df)
                true_params.append(true_theta)
                
        result_df = pd.concat(output)
        print("\n" + "="*60)
        print(result_df)
        return result_df, true_params


# ============================================================================
# Training Function
# ============================================================================

def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in PCE bound denominator
    num_outer_samples,  # N to estimate outer expectation
    lr,                 # learning rate
    gamma,              # learning rate scheduler decay
    T,                  # number of experiments
    intensity_min,      # minimum intensity
    intensity_max,      # maximum intensity
    # Prior hyperparameters
    a_loc,
    a_scale,
    b_scale,
    L_scale,
    ell_scale,
    H_scale,
    c1_scale,
    c2_scale,
    # Network architecture
    hidden_dim,
    encoding_dim,
    device,
    mlflow_experiment_name,
    design_network_type,  # "dad" or "static" or "random"
    adam_betas_wd=[0.9, 0.999, 0],
    mlflow_tracking_uri=None,
    mlflow_run_name=None,
):
    """
    Train MEP design network using DAD framework.
    """
    pyro.clear_param_store()
    seed = auto_seed(seed)
    *adam_betas, adam_weight_decay = adam_betas_wd

    ### Set up design network ###
    n = 1  # batch dimension
    encoder = EncoderNetwork((n, 1), (n, 1), hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim, (n, 1), intensity_min, intensity_max)
    
    if design_network_type == "static":
        design_net = BatchDesignBaseline(T, (n, 1)).to(device)
    elif design_network_type == "random":
        design_net = RandomDesignBaseline(T, (n, 1)).to(device)
        num_steps = 0  # no gradient steps needed
    elif design_network_type == "dad":
        # Initialize empty value to midpoint of intensity range
        empty_value = torch.ones(n, 1) * (intensity_min + intensity_max) / 2.0
        design_net = SetEquivariantDesignNetwork(
            encoder, emitter, empty_value=empty_value
        ).to(device)
    else:
        raise ValueError(f"design_network_type={design_network_type} not supported.")

    ### Set up MLflow logging ###
    if mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run(run_name=mlflow_run_name):
        ## Reproducibility
        mlflow.log_param("seed", seed)
        
        ## Model hyperparameters
        mlflow.log_param("num_experiments", T)
        mlflow.log_param("intensity_min", intensity_min)
        mlflow.log_param("intensity_max", intensity_max)
        mlflow.log_param("a_loc", a_loc)
        mlflow.log_param("a_scale", a_scale)
        mlflow.log_param("b_scale", b_scale)
        mlflow.log_param("L_scale", L_scale)
        mlflow.log_param("ell_scale", ell_scale)
        mlflow.log_param("H_scale", H_scale)
        mlflow.log_param("c1_scale", c1_scale)
        mlflow.log_param("c2_scale", c2_scale)
        
        ## Design network hyperparameters
        mlflow.log_param("design_network_type", design_network_type)
        if design_network_type == "dad":
            mlflow.log_param("hidden_dim", hidden_dim)
            mlflow.log_param("encoding_dim", encoding_dim)
        mlflow.log_param("num_inner_samples", num_inner_samples)
        mlflow.log_param("num_outer_samples", num_outer_samples)
        
        ## Optimizer hyperparameters
        mlflow.log_param("num_steps", num_steps)
        mlflow.log_param("lr", lr)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("adam_beta1", adam_betas[0])
        mlflow.log_param("adam_beta2", adam_betas[1])
        mlflow.log_param("adam_weight_decay", adam_weight_decay)

        ### Create model ###
        mep_model = MEPModel(
            design_net=design_net,
            a_loc=a_loc,
            a_scale=a_scale,
            b_scale=b_scale,
            L_scale=L_scale,
            ell_scale=ell_scale,
            H_scale=H_scale,
            c1_scale=c1_scale,
            c2_scale=c2_scale,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            T=T,
        )

        ### Set up optimizer ###
        optimizer = torch.optim.Adam
        scheduler = pyro.optim.ExponentialLR(
            {
                "optimizer": optimizer,
                "optim_args": {
                    "lr": lr,
                    "betas": adam_betas,
                    "weight_decay": adam_weight_decay,
                },
                "gamma": gamma,
            }
        )
        
        ### Set up loss (PCE bound) ###
        pce_loss = PriorContrastiveEstimation(num_outer_samples, num_inner_samples)

        ### Create OED object ###
        oed = OED(mep_model.model, scheduler, pce_loss)

        ### Optimize ###
        loss_history = []
        num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
        
        for i in num_steps_range:
            loss = oed.step()
            loss = torch_item(loss)
            loss_history.append(loss)
            
            # Log every 50 steps
            if i % 50 == 0:
                num_steps_range.set_description("Loss: {:.3f} ".format(loss))
                loss_eval = oed.evaluate_loss()
                mlflow.log_metric("loss", loss_eval, step=i)
                
            # Decrease learning rate every 1000 steps
            if i % 1000 == 0:
                scheduler.step()

        # Log final metrics
        if len(loss_history) == 0:
            # Random designs - no gradient updates
            loss = torch_item(pce_loss.differentiable_loss(mep_model.model))
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("loss_diff50", 0)
            mlflow.log_metric("loss_av50", loss)
        else:
            loss_diff50 = np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
            mlflow.log_metric("loss_diff50", loss_diff50)
            loss_av50 = np.mean(loss_history[-51:-1])
            mlflow.log_metric("loss_av50", loss_av50)

        mep_model.eval()
        
        # Store model as MLflow artifact
        print("Storing model to MLflow... ", end="")
        mlflow.pytorch.log_model(mep_model.cpu(), "model")
        ml_info = mlflow.active_run().info
        model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
        print(f"Model stored in {model_loc}. Done.")
        print(f"The experiment-id of this run is {ml_info.experiment_id}")

        return mep_model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design: MEP Recruitment Curve Estimation"
    )
    
    # Training parameters
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=1000, type=int)
    parser.add_argument("--num-outer-samples", default=1000, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--gamma", default=0.98, type=float)
    
    # Experiment parameters
    parser.add_argument("--num-experiments", default=30, type=int)
    parser.add_argument("--intensity-min", default=0.0, type=float)
    parser.add_argument("--intensity-max", default=100.0, type=float)
    
    # Prior hyperparameters
    parser.add_argument("--a-loc", default=50.0, type=float)
    parser.add_argument("--a-scale", default=50.0, type=float) 
    parser.add_argument("--b-scale", default=1.0, type=float)  
    parser.add_argument("--L-scale", default=0.1, type=float)
    parser.add_argument("--ell-scale", default=5.0, type=float)  
    parser.add_argument("--H-scale", default=5.0, type=float)    
    parser.add_argument("--c1-scale", default=5.0, type=float) 
    parser.add_argument("--c2-scale", default=0.5, type=float) 
    
    # Network architecture
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=8, type=int)
    parser.add_argument("--design-network-type", default="dad", type=str)
    parser.add_argument("--adam-betas-wd", nargs="+", default=[0.9, 0.999, 0])
    parser.add_argument(
        "--mlflow-experiment-name", default="mep_recruitment", type=str
    )
    parser.add_argument("--mlflow-tracking-uri", default=None, type=str,
                    help="MLflow tracking URI (e.g. file:/path/to/mlruns)")
    parser.add_argument("--mlflow-run-name", default=None, type=str,
                        help="Optional MLflow run name (e.g. dad_50k_123456)")

    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        T=args.num_experiments,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        a_loc=args.a_loc,
        a_scale=args.a_scale,
        b_scale=args.b_scale,
        L_scale=args.L_scale,
        ell_scale=args.ell_scale,
        H_scale=args.H_scale,
        c1_scale=args.c1_scale,
        c2_scale=args.c2_scale,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        device=args.device,
        mlflow_experiment_name=args.mlflow_experiment_name,
        design_network_type=args.design_network_type,
        adam_betas_wd=args.adam_betas_wd,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_run_name=args.mlflow_run_name,
    )