#!/usr/bin/env python3

import runpod
import os
from runpod.api import graphql

runpod.api_key = os.environ['RUNPOD_API_KEY']

REGISTRY_NAME = os.environ['REGISTRY_NAME']
ECR_TOKEN = os.environ['ECR_TOKEN']

print(f"Managing registry credentials: {REGISTRY_NAME}")

# Get existing credentials
query = """
query {
    myself {
        containerRegistryCreds {
            id
            name
        }
    }
}
"""

result = graphql.run_graphql_query(query)
creds = result["data"]["myself"]["containerRegistryCreds"]

found_cred = None
for cred in creds:
    if cred["name"] == REGISTRY_NAME:
        found_cred = cred
        break

if found_cred:
    # Update existing
    print(f"Updating credentials (ID: {found_cred['id']})")

    mutation = f"""
    mutation {{
        updateRegistryAuth(input: {{
            id: "{found_cred['id']}"
            username: "AWS"
            password: "{ECR_TOKEN}"
        }}) {{
            id
            name
        }}
    }}
    """

    result = graphql.run_graphql_query(mutation)
    registry_id = result["data"]["updateRegistryAuth"]["id"]
    print(f"✅ Updated: {registry_id}")
else:
    # Create new
    print("Creating new credentials")

    mutation = f"""
    mutation {{
        saveRegistryAuth(input: {{
            name: "{REGISTRY_NAME}"
            username: "AWS"
            password: "{ECR_TOKEN}"
        }}) {{
            id
            name
        }}
    }}
    """

    result = graphql.run_graphql_query(mutation)
    registry_id = result["data"]["saveRegistryAuth"]["id"]
    print(f"✅ Created: {registry_id}")

# Save output
with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    f.write(f"registry_id={registry_id}\n")
