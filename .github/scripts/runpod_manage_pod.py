#!/usr/bin/env python3

import runpod
from runpod.api import graphql
import os
import sys
import time

PRIVATE_TCP_PORT = 8000

runpod.api_key = os.environ['RUNPOD_API_KEY']

TEMPLATE_NAME = os.environ['TEMPLATE_NAME']
POD_NAME = os.environ['POD_NAME']
TEMPLATE_ID = os.environ['TEMPLATE_ID']
IMAGE_URI = os.environ['IMAGE_URI']
GPU_TYPE = os.environ['GPU_TYPE']
USE_SPOT = os.environ['USE_SPOT'] == 'true'
USE_NETWORK_VOLUME = os.environ['USE_NETWORK_VOLUME'] == 'true'
REPLACE = os.environ['REPLACE'] == 'true'
DATA_CENTER_ID = os.environ['DATA_CENTER_ID']
VOLUME_SIZE_GB = int(os.environ['VOLUME_SIZE_GB'])
VOLUME_MOUNT_PATH = os.environ['VOLUME_MOUNT_PATH']

if USE_NETWORK_VOLUME:
    NETWORK_VOLUME_ID = os.environ['NETWORK_VOLUME_ID']


def get_pod_url(pod_obj):
    """
    Extract pod URL from pod runtime information.
    Handles both HTTP proxy URLs and TCP direct connections.

    Returns:
        str: Pod URL (http://ip:port or https://proxy-url)
    """
    # Query pod details including runtime

    if not pod_obj.get("runtime") or not pod_obj["runtime"].get("ports"):
        print("⚠️  Warning: Pod has no runtime ports yet")
        return None

    ports = pod_obj["runtime"]["ports"]

    # Look for port 8000 (our service port)
    for port_info in ports:
        if port_info.get("privatePort") == PRIVATE_TCP_PORT:
            port_type = port_info.get("type", "").lower()

            if port_type == "http":
                # HTTP proxy URL
                pod_url = f"https://{pod_obj.get('id')}-{PRIVATE_TCP_PORT}.proxy.runpod.net"
                print(f"   Type: HTTP Proxy")
                print(f"   URL: {pod_url}")
                return pod_url

            elif port_type == "tcp":
                # TCP direct connection
                ip = port_info.get("ip")
                public_port = port_info.get("publicPort")
                is_public = port_info.get("isIpPublic", False)

                if not ip or not public_port:
                    print("⚠️  Warning: TCP port info incomplete")
                    return None

                pod_url = f"http://{ip}:{public_port}"
                print(f"   Type: TCP Direct")
                print(f"   IP: {ip} ({'Public' if is_public else 'Private'})")
                print(f"   Port: {public_port} (mapped from {PRIVATE_TCP_PORT})")
                print(f"   URL: {pod_url}")
                return pod_url

            else:
                print(f"⚠️  Unknown port type: {port_type}")
                return None

    # Port 8000 not found
    print(f"⚠️  Warning: Port {PRIVATE_TCP_PORT} not found in runtime ports")
    print(f"   Available ports: {[p.get('privatePort') for p in ports]}")
    return None


def manage_pod():
    print(f"Managing pod: {POD_NAME} ...")

    print("=" * 60)
    print("DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    print(f"Pod: {POD_NAME}")
    print(f"Template: {TEMPLATE_NAME}")
    print(f"Data Center: {DATA_CENTER_ID}")
    print(f"Image: {IMAGE_URI}")
    print(f"GPU: {GPU_TYPE}")
    print(f"Type: {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Replace: {REPLACE}")
    print("=" * 60)
    print()

    pods = runpod.get_pods()
    existing_pod = None

    for pod in pods:
        if pod.get("name") == POD_NAME:
            existing_pod = pod
            break

    if existing_pod:
        print(f"✅ Found existing pod:")
        print(f"   ID: {existing_pod['id']}")
        print(f"   Status: {existing_pod.get('desiredStatus', 'unknown')}")

        if REPLACE:
            print(f"Terminating: {existing_pod['id']}")
            print("✅ Network volume will be preserved")
            runpod.terminate_pod(existing_pod['id'])
            time.sleep(30)
        else:
            pod_id = existing_pod['id']
            pod_url = get_pod_url(existing_pod)
            print(f"⚠️  Pod already exists: {pod_id}")
            print(f"URL: {pod_url}")
            print("Skipping deployment (replace_existing_pod is false)")

            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"pod_id={pod_id}\n")
                f.write(f"pod_url={pod_url}\n")
                f.write(f"action=skipped\n")
            sys.exit(0)

    # Deploy new pod
    print(f"Deploying new pod in data center {DATA_CENTER_ID} ...")

    if USE_SPOT:
        mutation_name = "podRentInterruptable"
        spot_field = "bidPerGpu: 0.0"
    else:
        mutation_name = "podFindAndDeployOnDemand"
        spot_field = ""

    mutation = f"""
    mutation {{
        {mutation_name}(input: {{
            {spot_field}
            cloudType: SECURE
            gpuCount: 1
            gpuTypeId: "{GPU_TYPE}"
            name: "{POD_NAME}"
            templateId: "{TEMPLATE_ID}"
            dataCenterId: "{DATA_CENTER_ID}"
            volumeInGb: {VOLUME_SIZE_GB}
            volumeMountPath: "{VOLUME_MOUNT_PATH}"
        }}) {{
            id
            desiredStatus
            machine {{
                podHostId
            }}
        }}
    }}
    """

    if USE_NETWORK_VOLUME:
        mutation = f"""
      mutation {{
          {mutation_name}(input: {{
              {spot_field}
              cloudType: SECURE
              gpuCount: 1
              gpuTypeId: "{GPU_TYPE}"
              name: "{POD_NAME}"
              templateId: "{TEMPLATE_ID}"
              dataCenterId: "{DATA_CENTER_ID}"
              networkVolumeId: "{NETWORK_VOLUME_ID}"
          }}) {{
              id
              desiredStatus
              machine {{
                  podHostId
              }}
          }}
      }}
      """

    try:
        result = graphql.run_graphql_query(mutation)
        pod_data = result["data"][mutation_name]
        pod_id = pod_data["id"]
        pod_url = get_pod_url(pod_data)

        print(f"✅ Pod deployed:")
        print(f"   ID: {pod_id}")
        print(f"   Status: {pod_data.get('desiredStatus', 'unknown')}")
        print(f"   Datacenter: {DATA_CENTER_ID}")
        print(f"   URL: {pod_url}")

    except Exception as e:
        print(f"❌ Deployment failed: {e}")

        error_str = str(e).lower()
        if "not available" in error_str or "no longer any instances" in error_str:
            print(f"\n💡 GPU '{GPU_TYPE}' not available in {DATA_CENTER_ID}")
            print("   Suggestions:")
            print("   - Try different datacenter")
            print("   - Try different GPU type")
            print("   - Enable Spot Instance for more capacity")
        elif "network volume" in error_str:
            print(f"\n💡 Network volume issue")
            print("   Check that volume exists in same datacenter")

        raise

    # Wait for pod to be ready
    print("\nWaiting for pod to start...")
    for i in range(60):
        try:
            pod = runpod.get_pod(pod_id)
            if pod.get("runtime", {}).get("ports"):
                print("✅ Pod is ready!")
                break
        except:
            pass
        time.sleep(5)
    else:
        print("⚠️  Pod started but not fully ready yet")

    # Save outputs
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"pod_id={pod_id}\n")
        f.write(f"pod_url={pod_url}\n")
        f.write(f"action=deployed\n")
        f.write(f"template_id={TEMPLATE_ID}\n")
        f.write(f"datacenter={DATA_CENTER_ID}\n")
        if USE_NETWORK_VOLUME:
            f.write(f"network_volume_id={NETWORK_VOLUME_ID}\n")

    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Pod ID:         {pod_id}")
    print(f"Pod URL:        {pod_url}")
    print(f"Template ID:    {TEMPLATE_ID}")
    print(f"GPU Type:       {GPU_TYPE}")
    print(f"Instance Type:  {'Spot' if USE_SPOT else 'On-Demand'}")
    print(f"Datacenter:       {DATA_CENTER_ID}")
    print(f"Image:          {IMAGE_URI}")
    if USE_NETWORK_VOLUME:
        print(f"Network Volume:   {NETWORK_VOLUME_ID}")
    print("=" * 60)

    if USE_SPOT:
        print("\n⚠️  SPOT INSTANCE WARNING:")
        print("   - Pod may be interrupted if outbid")
        print("   - Save work frequently")


if __name__ == "__main__":
    manage_pod()