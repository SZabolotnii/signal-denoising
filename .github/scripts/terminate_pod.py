import os
import sys
import requests
from dotenv import load_dotenv

def terminate_pod():
    # Load .env if it exists (for local testing/debugging)
    load_dotenv()
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not found in environment.")
        return False
        
    pod_id = os.getenv("RUNPOD_POD_ID") or os.getenv("POD_ID")
    if not pod_id:
        print("RUNPOD_POD_ID not found in environment. Cannot terminate this pod automatically.")
        return False
        
    print(f"Terminating RunPod instance: {pod_id}...")
    
    url = f"https://api.runpod.io/v2/graphql?api_key={api_key}"
    
    # GraphQL mutation to terminate a pod
    query = """
    mutation terminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    
    variables = {
        "input": {
            "podId": pod_id
        }
    }
    
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        if response.status_code == 200:
            result = response.json()
            if "errors" in result:
                # Handle case where pod is already terminating or not found
                err_msg = str(result['errors'])
                if "not found" in err_msg.lower():
                    print(f"Pod {pod_id} not found, it might be already terminated.")
                    return True
                print(f"Error terminating pod: {result['errors']}")
                return False
            print(f"Pod {pod_id} termination requested successfully.")
            return True
        else:
            print(f"Failed to terminate pod. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error while calling RunPod API: {e}")
        return False

if __name__ == "__main__":
    if terminate_pod():
        sys.exit(0)
    else:
        sys.exit(1)