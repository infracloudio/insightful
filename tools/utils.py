import json
import re

def cleanupInputAndGetDictFromStr(input: str) -> dict:
    input = input.replace("\n", "")
    match = re.search(r'(\{.*\}|\[.*\])', input)
    cleaned_json_string = input

    if match:
        cleaned_json_string = match.group(0)
    
    formattedInput = cleaned_json_string
    if "True" in formattedInput and '"True"' not in formattedInput:
        formattedInput = formattedInput.replace("True", "\"True\"")
    if "False" in formattedInput and '"False"' not in formattedInput:
        formattedInput = formattedInput.replace("False", "\"False\"")
        
    try:
      params = json.loads(formattedInput)
      cleaned_params = {k: v for k, v in params.items() if v not in (0, None, "", "all", [], {})}
    except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON input: {e}")
        
    return cleaned_params