#!/usr/bin/env python3

import os
import runpod
from runpod.api import graphql

runpod.api_key = os.environ["RUNPOD_API_KEY"]

NETWORK_VOLUME_NAME = os.environ["NETWORK_VOLUME_NAME"]
NETWORK_VOLUME_SIZE_GB = int(os.environ["NETWORK_VOLUME_SIZE_GB"])
DATA_CENTER_ID = os.environ.get("DATA_CENTER_ID", "AUTO")
DEPLOY_SCOPE = os.environ.get("DEPLOY_SCOPE", "EU+US").upper()

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


def get_candidate_datacenters(preferred_dc: str, deploy_scope: str) -> list[str]:
    pool = EU_DCS + US_DCS + CA_DCS

    if preferred_dc and preferred_dc != "AUTO":
        if preferred_dc in pool:
            return [preferred_dc]
        raise RuntimeError(f"Unsupported DATA_CENTER_ID: {preferred_dc}")

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

    raise RuntimeError(f"Unsupported DEPLOY_SCOPE: {deploy_scope}")


def list_network_volumes():
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
    result = graphql.run_graphql_query(query)
    return result["data"]["myself"]["networkVolumes"]


def find_existing_volume(volumes, name: str, datacenter_ids: list[str]):
    for vol in volumes:
        if vol.get("name") == name and vol.get("dataCenterId") in datacenter_ids:
            return vol
    return None


def create_network_volume(name: str, size_gb: int, datacenter_id: str):
    mutation = f"""
    mutation {{
      createNetworkVolume(input: {{
          name: "{name}"
          size: {size_gb}
          dataCenterId: "{datacenter_id}"
      }}) {{
          id
          name
          size
          dataCenterId
      }}
    }}
    """
    result = graphql.run_graphql_query(mutation)
    return result["data"]["createNetworkVolume"]


def write_outputs(volume_id: str, volume_dc: str):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"network_volume_id={volume_id}\n")
        f.write(f"network_volume_datacenter={volume_dc}\n")


def main():
    candidate_dcs = get_candidate_datacenters(DATA_CENTER_ID, DEPLOY_SCOPE)

    print(f"Managing network volume: {NETWORK_VOLUME_NAME}")
    print(f"Requested data center: {DATA_CENTER_ID}")
    print(f"Deploy scope: {DEPLOY_SCOPE}")
    print(f"Candidate datacenters: {', '.join(candidate_dcs)}")

    volumes = list_network_volumes()

    existing = find_existing_volume(volumes, NETWORK_VOLUME_NAME, candidate_dcs)
    if existing:
        print("Found existing volume:")
        print(f"  ID: {existing['id']}")
        print(f"  Name: {existing['name']}")
        print(f"  Datacenter: {existing['dataCenterId']}")
        print(f"  Size: {existing['size']}GB")

        write_outputs(existing["id"], existing["dataCenterId"])
        return

    # Create in the first concrete datacenter from the candidate list
    selected_dc = candidate_dcs[0]
    print(f"Creating new network volume: {NETWORK_VOLUME_NAME} in {selected_dc}")

    created = create_network_volume(
        name=NETWORK_VOLUME_NAME,
        size_gb=NETWORK_VOLUME_SIZE_GB,
        datacenter_id=selected_dc,
    )

    print("Created network volume:")
    print(f"  ID: {created['id']}")
    print(f"  Name: {created['name']}")
    print(f"  Datacenter: {created['dataCenterId']}")
    print(f"  Size: {created['size']}GB")

    write_outputs(created["id"], created["dataCenterId"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed to manage network volume: {e}")
        raise