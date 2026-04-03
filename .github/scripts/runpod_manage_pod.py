#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional, List, Dict, Any

import runpod
from runpod.api import graphql

PRIVATE_TCP_PORT = 8000

runpod.api_key = os.environ['RUNPOD_API_KEY']

POD_NAME = os.environ["POD_NAME"]
TEMPLATE_NAME = os.environ["TEMPLATE_NAME"]
TEMPLATE_ID = os.environ["TEMPLATE_ID"]
IMAGE_URI = os.environ["IMAGE_URI"]
GPU_TYPE = os.environ["GPU_TYPE"]
DATA_CENTER_ID = os.environ.get("DATA_CENTER_ID", "AUTO")
USE_SPOT = os.environ.get("USE_SPOT", "false").lower() == "true"
USE_NETWORK_VOLUME = os.environ.get("USE_NETWORK_VOLUME", "false").lower() == "true"
REPLACE = os.environ.get("REPLACE", "false").lower() == "true"
VOLUME_SIZE_GB = int(os.environ.get("VOLUME_SIZE_GB", "30"))
VOLUME_MOUNT_PATH = os.environ.get("VOLUME_MOUNT_PATH", "/app/data")
NETWORK_VOLUME_ID = os.environ.get("NETWORK_VOLUME_ID")

# Optional grouping input if you add it later
DEPLOY_SCOPE = os.environ.get("DEPLOY_SCOPE", "EU+US").upper()

# ----------------------------
# Region candidates
# Keep these to known/current IDs from Runpod docs/examples.
# ----------------------------
EU_DCS = [
    "EU-NL-1",
    "EU-RO-1",
    "EU-CZ-1",
    "EUR-IS-1",
    "EUR-IS-3",
    "EUR-NO-1",
]

US_DCS = [
    "US-IL-1",
    "US-TX-3",
    "US-KS-2",
    "US-GA-2",
    "US-WA-1",
    "US-MO-2",
    "US-NC-1",
]

CA_DCS = [
    "CA-MTL-4",
    "CA-MTL-3",
]


def write_github_output(**kwargs: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def get_candidate_datacenters(
        preferred_dc: str,
        deploy_scope: str,
) -> List[str]:
    if preferred_dc and preferred_dc != "AUTO":
        pool = EU_DCS + US_DCS + CA_DCS
        if preferred_dc in pool:
            return [preferred_dc] + [dc for dc in pool if dc != preferred_dc]
        return [preferred_dc]

    if deploy_scope == "EU":
        return EU_DCS[:]
    if deploy_scope == "US":
        return US_DCS[:]
    if deploy_scope == "CA":
        return CA_DCS[:]
    if deploy_scope == "EU+US":
        return EU_DCS + US_DCS
    if deploy_scope == "ALL":
        return EU_DCS + US_DCS + CA_DCS

    return EU_DCS + US_DCS


def get_network_volume_datacenter(network_volume_id: str) -> str:
    query = f"""
    query {{
      myself {{
        networkVolumes {{
          id
          name
          dataCenterId
        }}
      }}
    }}
    """
    result = graphql.run_graphql_query(query)
    if result.get("errors"):
        raise RuntimeError(result["errors"][0]["message"])

    volumes = result["data"]["myself"]["networkVolumes"]
    for volume in volumes:
        if volume["id"] == network_volume_id:
            return volume["dataCenterId"]

    raise RuntimeError(f"Network volume not found: {network_volume_id}")


def get_pod_url(pod_obj: Dict[str, Any]) -> Optional[str]:
    runtime = pod_obj.get("runtime") or {}
    ports = runtime.get("ports") or []

    if not ports:
        print("Warning: Pod has no runtime ports yet")
        return None

    for port_info in ports:
        if port_info.get("privatePort") != PRIVATE_TCP_PORT:
            continue

        port_type = (port_info.get("type") or "").lower()

        if port_type == "http":
            pod_url = f"https://{pod_obj.get('id')}-{PRIVATE_TCP_PORT}.proxy.runpod.net"
            print(f"Type: HTTP Proxy")
            print(f"URL: {pod_url}")
            return pod_url

        if port_type == "tcp":
            ip = port_info.get("ip")
            public_port = port_info.get("publicPort")
            is_public = port_info.get("isIpPublic", False)

            if not ip or not public_port:
                print("Warning: TCP port info incomplete")
                return None

            pod_url = f"http://{ip}:{public_port}"
            print("Type: TCP Direct")
            print(f"IP: {ip} ({'Public' if is_public else 'Private'})")
            print(f"Port: {public_port} (mapped from {PRIVATE_TCP_PORT})")
            print(f"URL: {pod_url}")
            return pod_url

        print(f"Unknown port type: {port_type}")
        return None

    print(f"Warning: Port {PRIVATE_TCP_PORT} not found")
    print(f"Available ports: {[p.get('privatePort') for p in ports]}")
    return None


def wait_for_pod_runtime(pod_id: str, timeout_sec: int = 300, poll_sec: int = 5) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_pod = None

    while time.time() < deadline:
        try:
            pod = runpod.get_pod(pod_id)
            last_pod = pod
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            if ports:
                return pod
        except Exception:
            pass

        time.sleep(poll_sec)

    return last_pod or {}


def terminate_existing_pod_if_needed() -> Optional[Dict[str, Any]]:
    pods = runpod.get_pods()
    for pod in pods:
        if pod.get("name") != POD_NAME:
            continue

        print("Found existing pod:")
        print(f"  ID: {pod['id']}")
        print(f"  Status: {pod.get('desiredStatus', 'unknown')}")

        if REPLACE:
            print(f"Terminating existing pod: {pod['id']}")
            runpod.terminate_pod(pod["id"])
            time.sleep(20)
            return None

        pod_id = pod["id"]
        full_pod = runpod.get_pod(pod_id)
        pod_url = get_pod_url(full_pod)

        print(f"Pod already exists: {pod_id}")
        print(f"URL: {pod_url}")
        print("Skipping deployment because REPLACE=false")

        write_github_output(
            pod_id=pod_id,
            pod_url=pod_url or "",
            action="skipped",
            datacenter=pod.get("dataCenterId", ""),
        )
        sys.exit(0)

    return None


def build_mutation_for_dc(dc: str) -> str:
    mutation_name = "podRentInterruptable" if USE_SPOT else "podFindAndDeployOnDemand"
    bid_line = "bidPerGpu: 0.0" if USE_SPOT else ""

    if USE_NETWORK_VOLUME:
        if not NETWORK_VOLUME_ID:
            raise RuntimeError("USE_NETWORK_VOLUME=true but NETWORK_VOLUME_ID is missing")

        input_body = f"""
            {bid_line}
            cloudType: SECURE
            gpuCount: 1
            gpuTypeId: "{GPU_TYPE}"
            name: "{POD_NAME}"
            templateId: "{TEMPLATE_ID}"
            dataCenterId: "{dc}"
            networkVolumeId: "{NETWORK_VOLUME_ID}"
        """
    else:
        input_body = f"""
            {bid_line}
            cloudType: SECURE
            gpuCount: 1
            gpuTypeId: "{GPU_TYPE}"
            name: "{POD_NAME}"
            templateId: "{TEMPLATE_ID}"
            dataCenterId: "{dc}"
            volumeInGb: {VOLUME_SIZE_GB}
            volumeMountPath: "{VOLUME_MOUNT_PATH}"
        """

    return f"""
    mutation {{
      {mutation_name}(input: {{
        {input_body}
      }}) {{
        id
        desiredStatus
        dataCenterId
      }}
    }}
    """


def deploy_with_fallback(dcs_to_try: List[str]) -> Dict[str, Any]:
    mutation_name = "podRentInterruptable" if USE_SPOT else "podFindAndDeployOnDemand"
    last_error = None

    for dc in dcs_to_try:
        print(f"Trying datacenter: {dc}")

        mutation = build_mutation_for_dc(dc)

        try:
            result = graphql.run_graphql_query(mutation)

            if result.get("errors"):
                raise RuntimeError(result["errors"][0]["message"])

            pod_data = result["data"][mutation_name]
            if not pod_data or not pod_data.get("id"):
                raise RuntimeError(f"Create pod returned empty data for {dc}")

            print(f"Deployment initiated in {dc}")
            return {
                "pod_id": pod_data["id"],
                "final_dc": pod_data.get("dataCenterId") or dc,
                "desired_status": pod_data.get("desiredStatus", "unknown"),
            }

        except Exception as e:
            last_error = e
            msg = str(e).lower()
            print(f"Failed in {dc}: {e}")

            retryable_markers = [
                "not available",
                "no longer any instances",
                "insufficient",
                "capacity",
                "unavailable",
            ]

            if any(marker in msg for marker in retryable_markers):
                continue

            if "network volume" in msg:
                raise RuntimeError(
                    f"Network volume issue in {dc}: {e}. "
                    "Network volumes are datacenter-specific."
                ) from e

            continue

    raise RuntimeError(f"All deployment attempts failed. Last error: {last_error}")


def manage_pod() -> None:
    print("=" * 60)
    print("DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    print(f"Pod: {POD_NAME}")
    print(f"Template: {TEMPLATE_NAME}")
    print(f"Requested DC: {DATA_CENTER_ID}")
    print(f"Scope: {DEPLOY_SCOPE}")
    print(f"Image: {IMAGE_URI}")
    print(f"GPU: {GPU_TYPE}")
    print(f"Type: {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Replace: {REPLACE}")
    print(f"Use network volume: {USE_NETWORK_VOLUME}")
    print("=" * 60)

    terminate_existing_pod_if_needed()

    if USE_NETWORK_VOLUME:
        volume_dc = get_network_volume_datacenter(NETWORK_VOLUME_ID)
        dcs_to_try = [volume_dc]
        print(f"Network volume {NETWORK_VOLUME_ID} is pinned to {volume_dc}")
    else:
        dcs_to_try = get_candidate_datacenters(DATA_CENTER_ID, DEPLOY_SCOPE)

    print(f"Candidate datacenters: {', '.join(dcs_to_try)}")

    deployment = deploy_with_fallback(dcs_to_try)
    pod_id = deployment["pod_id"]
    final_dc = deployment["final_dc"]

    print("Waiting for pod runtime...")
    full_pod = wait_for_pod_runtime(pod_id)

    pod_url = get_pod_url(full_pod) if full_pod else None

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Pod ID:        {pod_id}")
    print(f"Pod URL:       {pod_url}")
    print(f"Template ID:   {TEMPLATE_ID}")
    print(f"GPU Type:      {GPU_TYPE}")
    print(f"Instance Type: {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Datacenter:    {final_dc}")
    print(f"Image:         {IMAGE_URI}")
    if USE_NETWORK_VOLUME:
        print(f"Network Volume:{NETWORK_VOLUME_ID}")
    print("=" * 60)

    write_github_output(
        pod_id=pod_id,
        pod_url=pod_url or "",
        action="deployed",
        template_id=TEMPLATE_ID,
        datacenter=final_dc,
        network_volume_id=NETWORK_VOLUME_ID or "",
    )

    if USE_SPOT:
        print("\nSPOT WARNING:")
        print("- Pod may be interrupted")
        print("- Save work frequently")


if __name__ == "__main__":
    manage_pod()
