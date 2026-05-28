"""Unified generate command for RBG deployment artifacts.

Produces a deployable RBG YAML with optional planner role and profiling
ConfigMap. Supports three profiling data sources:
- configmap: reference an existing ConfigMap in the cluster
- json: embed local JSON files into a new ConfigMap
- auto: run the profiling pipeline to collect data automatically
"""

import asyncio
import json
import os
import sys

import click
import yaml

from inference_ext_cli.profile.config_modifiers import CONFIG_MODIFIERS
from inference_ext_cli.profile.defaults import DEFAULT_ENGINE_PORT
from inference_ext_cli.profile.parallelization_mapping import SubComponentType


def _build_planner_role(
    planner_image: str,
    profiling_configmap: str,
    prometheus_endpoint: str,
    rbg_name: str,
    namespace: str,
    model_name: str,
    prefill_role: str,
    decode_role: str,
    metric_source: str,
    ttft_sla: float,
    itl_sla: float,
    adjustment_interval: int,
    max_gpu_budget: int,
    min_replicas: int,
    prefill_engine_num_gpu: int,
    decode_engine_num_gpu: int,
    load_predictor: str,
    planner_prometheus_port: int,
) -> dict:
    """Build the planner role spec."""
    env_vars = [
        {"name": "RBG_NAME", "value": rbg_name},
        {"name": "RBG_NAMESPACE", "value": namespace},
        {"name": "PROMETHEUS_ENDPOINT", "value": prometheus_endpoint},
        {"name": "MODEL_NAME", "value": model_name},
        {"name": "PREFILL_ROLE_NAME", "value": prefill_role},
        {"name": "DECODE_ROLE_NAME", "value": decode_role},
        {"name": "METRIC_SOURCE", "value": metric_source},
        {"name": "TTFT_SLA", "value": str(ttft_sla)},
        {"name": "ITL_SLA", "value": str(itl_sla)},
        {"name": "ADJUSTMENT_INTERVAL", "value": str(adjustment_interval)},
        {"name": "MAX_GPU_BUDGET", "value": str(max_gpu_budget)},
        {"name": "MIN_REPLICAS", "value": str(min_replicas)},
        {"name": "PREFILL_ENGINE_NUM_GPU", "value": str(prefill_engine_num_gpu)},
        {"name": "DECODE_ENGINE_NUM_GPU", "value": str(decode_engine_num_gpu)},
        {"name": "LOAD_PREDICTOR", "value": load_predictor},
        {"name": "PROFILE_RESULTS_DIR", "value": "/etc/rbg-planner/profiling"},
        {"name": "PLANNER_PROMETHEUS_PORT", "value": str(planner_prometheus_port)},
    ]

    container = {
        "name": "rbg-planner",
        "image": planner_image,
        "env": env_vars,
        "volumeMounts": [
            {
                "name": "profiling-data",
                "mountPath": "/etc/rbg-planner/profiling",
                "readOnly": True,
            }
        ],
        "resources": {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
    }

    if planner_prometheus_port > 0:
        container["ports"] = [
            {"containerPort": planner_prometheus_port, "name": "metrics"}
        ]

    return {
        "name": "planner",
        "replicas": 1,
        "standalonePattern": {
            "template": {
                "spec": {
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "profiling-data",
                            "configMap": {"name": profiling_configmap},
                        }
                    ],
                }
            }
        },
    }


def _generate_rbg_from_scratch(
    engine: str,
    model: str,
    engine_image: str,
    prefill_tp: int,
    decode_tp: int,
    port: int = DEFAULT_ENGINE_PORT,
    is_moe: bool = False,
) -> dict:
    """Generate a full PD-disaggregated RBG spec from scratch."""
    config_modifier = CONFIG_MODIFIERS[engine]

    prefill_spec = config_modifier.generate_rbg_spec(
        model=model, image=engine_image, num_gpus=prefill_tp,
        port=port, phase=SubComponentType.PREFILL, is_moe=is_moe,
    )
    decode_spec = config_modifier.generate_rbg_spec(
        model=model, image=engine_image, num_gpus=decode_tp,
        port=port, phase=SubComponentType.DECODE, is_moe=is_moe,
    )

    # Rename roles
    prefill_role = prefill_spec["roles"][0]
    prefill_role["name"] = "prefill"
    decode_role = decode_spec["roles"][0]
    decode_role["name"] = "decode"

    return {
        "apiVersion": "workloads.x-k8s.io/v1alpha2",
        "kind": "RoleBasedGroup",
        "metadata": {
            "name": f"{engine}-pd-inference",
            "namespace": "default",
        },
        "spec": {
            "roles": [prefill_role, decode_role],
        },
    }


def _build_configmap(
    prefill_dict: dict, decode_dict: dict, name: str, namespace: str,
) -> dict:
    """Build a ConfigMap resource from profiling data dicts."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "data": {
            "prefill_raw_data.json": json.dumps(prefill_dict, indent=2),
            "decode_raw_data.json": json.dumps(decode_dict, indent=2),
        },
    }


@click.command("generate")
# Input source (one of)
@click.option("--rbg-yaml", type=click.Path(exists=True), help="Existing RBG YAML to modify")
@click.option("--engine", type=click.Choice(["sglang", "vllm"]), help="Engine type (for from-scratch)")
@click.option("--model", help="HuggingFace model ID (for from-scratch)")
@click.option("--engine-image", help="Engine container image (for from-scratch)")
@click.option("--prefill-tp", default=1, type=int, help="Prefill TP size (from-scratch)")
@click.option("--decode-tp", default=1, type=int, help="Decode TP size (from-scratch)")
# Planner options
@click.option("--enable-planner", is_flag=True, help="Add planner role to the RBG")
@click.option("--planner-image", help="Planner container image")
@click.option("--model-name", help="Model name for Prometheus metric filtering")
@click.option("--profiling-source", type=click.Choice(["configmap", "json", "auto"]), help="Profiling data source")
@click.option("--profiling-configmap", help="Existing ConfigMap name (source=configmap)")
@click.option("--prefill-json", type=click.Path(exists=True), help="Prefill JSON file (source=json)")
@click.option("--decode-json", type=click.Path(exists=True), help="Decode JSON file (source=json)")
@click.option("--prometheus-endpoint", default="http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090", help="Prometheus endpoint")
@click.option("--metric-source", default="sglang", type=click.Choice(["sglang", "vllm", "patio"]), help="Metric source")
@click.option("--ttft-sla", default=500.0, type=float, help="TTFT SLA target (ms)")
@click.option("--itl-sla", default=50.0, type=float, help="ITL SLA target (ms)")
@click.option("--adjustment-interval", default=180, type=int, help="Scaling interval (s)")
@click.option("--max-gpu-budget", default=8, type=int, help="Max GPU budget")
@click.option("--min-replicas", default=1, type=int, help="Min replicas per role")
@click.option("--prefill-engine-num-gpu", default=1, type=int, help="GPUs per prefill engine")
@click.option("--decode-engine-num-gpu", default=1, type=int, help="GPUs per decode engine")
@click.option("--load-predictor", default="arima", type=click.Choice(["constant", "arima", "prophet"]), help="Load predictor")
@click.option("--planner-prometheus-port", default=9091, type=int, help="Planner metrics port (0=disable)")
# Auto-profiling options
@click.option("--namespace", default="default", help="K8s namespace (for auto profiling)")
@click.option("--min-gpus", default=1, type=int, help="Min GPUs to sweep (auto)")
@click.option("--max-gpus", default=8, type=int, help="Max GPUs to sweep (auto)")
@click.option("--isl", default=3000, type=int, help="Target ISL (auto)")
@click.option("--osl", default=500, type=int, help="Target OSL (auto)")
@click.option("--max-context-length", default=None, type=int, help="Max context length (auto)")
@click.option("--prefill-interpolation-granularity", default=16, type=int, help="Prefill sweep points (auto)")
@click.option("--decode-interpolation-granularity", default=6, type=int, help="Decode sweep points (auto)")
@click.option("--deployment-timeout", default=1800, type=int, help="Deployment timeout (auto)")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code (auto)")
# Output
@click.option("--output-dir", "-o", default="./output", help="Output directory")
def generate_command(
    rbg_yaml, engine, model, engine_image, prefill_tp, decode_tp,
    enable_planner, planner_image, model_name, profiling_source,
    profiling_configmap, prefill_json, decode_json,
    prometheus_endpoint, metric_source, ttft_sla, itl_sla,
    adjustment_interval, max_gpu_budget, min_replicas,
    prefill_engine_num_gpu, decode_engine_num_gpu, load_predictor,
    planner_prometheus_port,
    namespace, min_gpus, max_gpus, isl, osl, max_context_length,
    prefill_interpolation_granularity, decode_interpolation_granularity,
    deployment_timeout, trust_remote_code,
    output_dir,
):
    """Generate RBG deployment YAML with optional planner and profiling ConfigMap.

    Two input modes:

    1. --rbg-yaml: Modify an existing RBG YAML (add planner role)

    2. --engine/--model/--engine-image: Generate a full PD-disaggregated RBG from scratch

    When --enable-planner is set, profiling data source must be specified:

    \b
    - configmap: reference an existing ConfigMap (--profiling-configmap)
    - json: create ConfigMap from local files (--prefill-json, --decode-json)
    - auto: run profiling pipeline to collect data (requires --engine-image, --namespace)
    """
    # --- Validate inputs ---
    if not rbg_yaml and not engine:
        click.echo("Error: provide either --rbg-yaml or --engine (for from-scratch generation)", err=True)
        sys.exit(1)

    if not rbg_yaml:
        if not model or not engine_image:
            click.echo("Error: --model and --engine-image are required for from-scratch generation", err=True)
            sys.exit(1)

    if enable_planner:
        if not planner_image:
            click.echo("Error: --planner-image is required with --enable-planner", err=True)
            sys.exit(1)
        if not model_name:
            click.echo("Error: --model-name is required with --enable-planner", err=True)
            sys.exit(1)
        if not profiling_source:
            click.echo("Error: --profiling-source is required with --enable-planner", err=True)
            sys.exit(1)

        if profiling_source == "configmap" and not profiling_configmap:
            click.echo("Error: --profiling-configmap is required with --profiling-source=configmap", err=True)
            sys.exit(1)
        if profiling_source == "json" and (not prefill_json or not decode_json):
            click.echo("Error: --prefill-json and --decode-json are required with --profiling-source=json", err=True)
            sys.exit(1)
        if profiling_source == "auto":
            if not engine_image:
                click.echo("Error: --engine-image is required with --profiling-source=auto", err=True)
                sys.exit(1)
            if not engine:
                # Try to infer engine from RBG YAML or require explicit
                click.echo("Error: --engine is required with --profiling-source=auto", err=True)
                sys.exit(1)

    # --- Build or load the RBG ---
    if rbg_yaml:
        with open(rbg_yaml, "r") as f:
            rbg = yaml.safe_load(f)
        if not rbg:
            click.echo("Error: empty or invalid RBG YAML", err=True)
            sys.exit(1)
    else:
        rbg = _generate_rbg_from_scratch(
            engine=engine, model=model, engine_image=engine_image,
            prefill_tp=prefill_tp, decode_tp=decode_tp,
        )

    rbg_name = rbg.get("metadata", {}).get("name", "")
    rbg_namespace = rbg.get("metadata", {}).get("namespace", "default")

    # --- Handle profiling data ---
    configmap_name = f"{rbg_name}-profiling"
    configmap_dict = None

    if enable_planner:
        if profiling_source == "configmap":
            configmap_name = profiling_configmap
        elif profiling_source == "json":
            with open(prefill_json, "r") as f:
                prefill_dict = json.load(f)
            with open(decode_json, "r") as f:
                decode_dict = json.load(f)
            configmap_dict = _build_configmap(
                prefill_dict, decode_dict, configmap_name, rbg_namespace,
            )
        elif profiling_source == "auto":
            import logging
            from inference_ext_cli.profile.command import run_profile

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            profile_output_dir = os.path.join(output_dir, "profiling-artifacts")
            asyncio.run(run_profile(
                engine=engine,
                model=model_name,
                engine_image=engine_image,
                namespace=namespace,
                min_gpus=min_gpus,
                max_gpus=max_gpus,
                isl=isl,
                osl=osl,
                ttft_sla=ttft_sla,
                itl_sla=itl_sla,
                max_context_length=max_context_length,
                prefill_interpolation_granularity=prefill_interpolation_granularity,
                decode_interpolation_granularity=decode_interpolation_granularity,
                output_dir=profile_output_dir,
                configmap_name=configmap_name,
                configmap_namespace=rbg_namespace,
                num_gpus_per_node=max_gpus,
                deployment_timeout=deployment_timeout,
                dry_run=False,
                trust_remote_code=trust_remote_code,
            ))

            # Load the generated profiling data
            prefill_json_path = os.path.join(profile_output_dir, "prefill_raw_data.json")
            decode_json_path = os.path.join(profile_output_dir, "decode_raw_data.json")
            with open(prefill_json_path, "r") as f:
                prefill_dict = json.load(f)
            with open(decode_json_path, "r") as f:
                decode_dict = json.load(f)
            configmap_dict = _build_configmap(
                prefill_dict, decode_dict, configmap_name, rbg_namespace,
            )

    # --- Inject planner role ---
    if enable_planner:
        planner_role = _build_planner_role(
            planner_image=planner_image,
            profiling_configmap=configmap_name,
            prometheus_endpoint=prometheus_endpoint,
            rbg_name=rbg_name,
            namespace=rbg_namespace,
            model_name=model_name,
            prefill_role="prefill",
            decode_role="decode",
            metric_source=metric_source,
            ttft_sla=ttft_sla,
            itl_sla=itl_sla,
            adjustment_interval=adjustment_interval,
            max_gpu_budget=max_gpu_budget,
            min_replicas=min_replicas,
            prefill_engine_num_gpu=prefill_engine_num_gpu,
            decode_engine_num_gpu=decode_engine_num_gpu,
            load_predictor=load_predictor,
            planner_prometheus_port=planner_prometheus_port,
        )

        roles = rbg.get("spec", {}).get("roles", [])
        existing_idx = next(
            (i for i, r in enumerate(roles) if r.get("name") == "planner"), None
        )
        if existing_idx is not None:
            roles[existing_idx] = planner_role
        else:
            roles.append(planner_role)
        rbg.setdefault("spec", {})["roles"] = roles

    # --- Write output ---
    os.makedirs(output_dir, exist_ok=True)

    rbg_output_path = os.path.join(output_dir, "rbg.yaml")
    with open(rbg_output_path, "w") as f:
        yaml.dump(rbg, f, default_flow_style=False, sort_keys=False)
    click.echo(f"RBG YAML written to {rbg_output_path}")

    if configmap_dict:
        cm_output_path = os.path.join(output_dir, "profiling-configmap.yaml")
        with open(cm_output_path, "w") as f:
            yaml.dump(configmap_dict, f, default_flow_style=False, sort_keys=False)
        click.echo(f"Profiling ConfigMap written to {cm_output_path}")
