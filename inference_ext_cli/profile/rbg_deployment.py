"""RBG deployment client for profiling.

Creates, waits for, and deletes temporary RoleBasedGroup deployments
used during the profiling pipeline. Replaces dynamo's DynamoDeploymentClient.
"""

import asyncio
import logging
import subprocess
import time
import uuid
from typing import Optional

import kubernetes_asyncio as kubernetes
from kubernetes_asyncio import client, config

from inference_ext_cli.profile.defaults import (
    DEFAULT_DEPLOYMENT_TIMEOUT,
    DEFAULT_ENGINE_PORT,
    RBG_GROUP,
    RBG_PLURAL,
    RBG_VERSION,
)

logger = logging.getLogger(__name__)


class RBGDeploymentClient:
    """Manages temporary RBG deployments for profiling."""

    def __init__(
        self,
        namespace: str,
        engine: str = "sglang",
        port: int = DEFAULT_ENGINE_PORT,
        base_log_dir: Optional[str] = None,
    ):
        self.namespace = namespace
        self.engine = engine
        self.port = port
        self.base_log_dir = base_log_dir

        self.rbg_name: Optional[str] = None
        self.role_name: str = "engine"
        self.k8s_client = None
        self.custom_api = None
        self.core_api = None
        self.port_forward_process: Optional[subprocess.Popen] = None

    async def _init_kubernetes(self):
        """Initialize kubernetes async client."""
        if self.k8s_client is not None:
            return

        try:
            config.load_incluster_config()
        except kubernetes.config.ConfigException:
            await config.load_kube_config()

        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

    async def create_deployment(self, rbg_spec: dict) -> str:
        """Create a temporary RBG with the given spec.

        Args:
            rbg_spec: The spec.roles portion of the RBG (output of config_modifier.generate_rbg_spec())

        Returns:
            The generated RBG name.
        """
        await self._init_kubernetes()

        suffix = str(uuid.uuid4())[:6]
        self.rbg_name = f"profiling-{self.engine}-{suffix}"

        body = {
            "apiVersion": f"{RBG_GROUP}/{RBG_VERSION}",
            "kind": "RoleBasedGroup",
            "metadata": {
                "name": self.rbg_name,
                "namespace": self.namespace,
            },
            "spec": rbg_spec,
        }

        logger.info(f"Creating RBG {self.rbg_name} in namespace {self.namespace}")

        try:
            await self.custom_api.create_namespaced_custom_object(
                group=RBG_GROUP,
                version=RBG_VERSION,
                namespace=self.namespace,
                plural=RBG_PLURAL,
                body=body,
            )
            logger.info(f"Successfully created RBG {self.rbg_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:
                logger.warning(f"RBG {self.rbg_name} already exists")
            else:
                logger.error(f"Failed to create RBG {self.rbg_name}: {e}")
                raise

        return self.rbg_name

    async def wait_for_ready(self, timeout: int = DEFAULT_DEPLOYMENT_TIMEOUT):
        """Wait for RBG status.conditions[Ready]=True.

        Args:
            timeout: Maximum wait time in seconds.

        Raises:
            TimeoutError: If the RBG doesn't become ready within timeout.
        """
        assert self.rbg_name, "No RBG created yet"
        start_time = time.time()
        check_interval = 15

        logger.info(f"Waiting for RBG {self.rbg_name} to become ready (timeout={timeout}s)...")

        while (time.time() - start_time) < timeout:
            try:
                rbg = await self.custom_api.get_namespaced_custom_object(
                    group=RBG_GROUP,
                    version=RBG_VERSION,
                    namespace=self.namespace,
                    plural=RBG_PLURAL,
                    name=self.rbg_name,
                )

                conditions = rbg.get("status", {}).get("conditions", [])
                for cond in conditions:
                    if cond.get("type") == "Ready" and cond.get("status") == "True":
                        elapsed = time.time() - start_time
                        logger.info(f"RBG {self.rbg_name} ready after {elapsed:.1f}s")
                        return

                # Log current state for debugging
                role_statuses = rbg.get("status", {}).get("roleStatuses", {})
                logger.debug(f"RBG {self.rbg_name} not ready yet. Roles: {role_statuses}")

            except kubernetes.client.rest.ApiException as e:
                logger.debug(f"API error while checking RBG status: {e.status}")

            await asyncio.sleep(check_interval)

        raise TimeoutError(
            f"RBG {self.rbg_name} failed to become ready within {timeout}s"
        )

    def get_service_url(self) -> str:
        """Get the K8s headless service URL for the engine.

        RBG creates services with pattern: s-{rbg-name}-{role-name}
        """
        assert self.rbg_name, "No RBG created yet"
        service_name = f"s-{self.rbg_name}-{self.role_name}"
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:{self.port}"

    def port_forward(self, local_port: Optional[int] = None) -> str:
        """Port forward the engine service to localhost.

        Args:
            local_port: Local port (defaults to self.port).

        Returns:
            Base URL string like "http://localhost:8000".
        """
        assert self.rbg_name, "No RBG created yet"

        if local_port is None:
            local_port = self.port

        service_name = f"s-{self.rbg_name}-{self.role_name}"
        cmd = [
            "kubectl", "port-forward",
            f"svc/{service_name}",
            f"{local_port}:{self.port}",
            "-n", self.namespace,
        ]

        logger.info(f"Starting port forward: {' '.join(cmd)}")
        self.port_forward_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Wait for port forward to establish
        return f"http://localhost:{local_port}"

    def stop_port_forward(self):
        """Stop port-forward subprocess."""
        if self.port_forward_process:
            self.port_forward_process.terminate()
            try:
                self.port_forward_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.port_forward_process.kill()
                self.port_forward_process.wait()
            self.port_forward_process = None

    async def get_pod_logs(self) -> str:
        """Get logs from the engine pod.

        Returns:
            Concatenated log text from all engine pods.
        """
        assert self.rbg_name, "No RBG created yet"
        await self._init_kubernetes()

        # RBG labels pods with role-based selectors
        label_selector = f"rbg.workloads.x-k8s.io/name={self.rbg_name},rbg.workloads.x-k8s.io/role={self.role_name}"

        pods = await self.core_api.list_namespaced_pod(
            namespace=self.namespace, label_selector=label_selector,
        )

        logs = []
        for pod in pods.items:
            try:
                pod_log = await self.core_api.read_namespaced_pod_log(
                    name=pod.metadata.name, namespace=self.namespace,
                )
                logs.append(pod_log)
            except kubernetes.client.rest.ApiException as e:
                logger.warning(f"Failed to get logs from pod {pod.metadata.name}: {e}")

        return "\n".join(logs)

    async def delete_deployment(self):
        """Delete the temporary RBG."""
        if not self.rbg_name:
            return

        self.stop_port_forward()

        try:
            await self._init_kubernetes()
            await self.custom_api.delete_namespaced_custom_object(
                group=RBG_GROUP,
                version=RBG_VERSION,
                namespace=self.namespace,
                plural=RBG_PLURAL,
                name=self.rbg_name,
            )
            logger.info(f"Deleted RBG {self.rbg_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete RBG {self.rbg_name}: {e}")
                raise
            logger.debug(f"RBG {self.rbg_name} already deleted")
        finally:
            if self.k8s_client:
                await self.k8s_client.close()
                self.k8s_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.delete_deployment()
        return False
