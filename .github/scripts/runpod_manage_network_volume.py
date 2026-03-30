#!/usr/bin/env python3

import runpod
from runpod.api import graphql
import os

runpod.api_key = os.environ['RUNPOD_API_KEY']

NETWORK_VOLUME_NAME = os.environ['NETWORK_VOLUME_NAME']
NETWORK_VOLUME_SIZE_GB = os.environ['NETWORK_VOLUME_SIZE_GB']
DATA_CENTER_ID = os.environ['DATA_CENTER_ID']

network_volume_id = None

print(f"Managing network volume {NETWORK_VOLUME_NAME} in data center {DATA_CENTER_ID} ...")

query = """
query {
    myself {
        networkVolumes {
            id
            name
            size
            dataCenterId
        }
    }
}
"""

try:
    result = graphql.run_graphql_query(query)
    volumes = result["data"]["myself"]["networkVolumes"]

    # Filter manually by name and datacenter
    network_volume_id = None
    for vol in volumes:
        if vol.get("name") == NETWORK_VOLUME_NAME and vol.get("dataCenterId") == DATA_CENTER_ID:
            network_volume_id = vol["id"]
            print(f"✅ Found existing volume:")
            print(f"   ID: {network_volume_id}")
            print(f"   Name: {vol['name']}")
            print(f"   Datacenter: {vol['dataCenterId']}")
            print(f"   Size: {vol['size']}GB")
            break

    if not network_volume_id:
        print(f"Creating new network volume: {NETWORK_VOLUME_NAME}")

        mutation = f"""
        mutation {{
          createNetworkVolume(input: {{
              name: "{NETWORK_VOLUME_NAME}"
              size: {NETWORK_VOLUME_SIZE_GB}
              dataCenterId: "{DATA_CENTER_ID}"
          }}) {{
              id
              name
              size
              dataCenterId
          }}
        }}
        """

        result = graphql.run_graphql_query(mutation)
        nv_data = result["data"]["createNetworkVolume"]
        network_volume_id = nv_data["id"]

        print(f"✅ Created network volume: {network_volume_id}")
        print(f"   Name: {NETWORK_VOLUME_NAME}")
        print(f"   Size: {nv_data['size']}GB")
        print(f"   Datacenter: {nv_data['dataCenterId']}")

    # Save output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"network_volume_id={network_volume_id}\n")

except Exception as e:
    print(f"❌ Failed to manage network volume: {e}")
    raise

