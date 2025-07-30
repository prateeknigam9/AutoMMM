import yaml
import json
from pydantic import BaseModel
import os
def load_prompt_config(path: str, key: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)[key]

def tools_to_action_prompt(tool_list:list):
    action = []
    tool_names = []
    for id, tool in enumerate(tool_list):
        tool_name = tool.name
        tool_desc = tool.description
        tool_args = tool.args
        action.append(f"tool_id: {id+1}\ntool_name: {tool_name}\ntool_description : {tool_desc}\ntool_arguments: {tool_args}\n")
        tool_names.append(tool_name)
    return "\n".join(action), tool_names


def resolve_ref(ref, definitions):
    """Resolve a $ref in the schema using the provided definitions."""
    ref_path = ref.split('/')[-1]
    return definitions.get(ref_path, {})

def extract_properties(schema, definitions):
    """Extract and flatten properties from the schema, resolving any $ref."""
    result = {}

    for prop, val in schema.get('properties', {}).items():
        if val.get('type') == 'array' and '$ref' in val.get('items', {}):
            # Handle arrays of objects defined via $ref
            ref_schema = resolve_ref(val['items']['$ref'], definitions)
            result[prop] = {k: {'type': v['type']} for k, v in ref_schema.get('properties', {}).items()}
        elif '$ref' in val:
            # Handle direct $ref to a schema
            ref_schema = resolve_ref(val['$ref'], definitions)
            result[prop] = {k: {'type': v['type']} for k, v in ref_schema.get('properties', {}).items()}
        else:
            # Direct property
            result[prop] = {'type': val.get('type', 'unknown')}
    return result


def print_model_schema(model: type[BaseModel], indent = 4):
    schema_dict = model.model_json_schema()
    schema = extract_properties(schema_dict, schema_dict.get('$defs', {}))
    return json.dumps(schema, indent=indent)

def save_in_memory(file_name, to_save, desc:str=''):
    path = f"memory/{file_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(to_save, dict):
            json.dump(to_save, f, ensure_ascii=False, indent=4)
        else:
            f.write(str(to_save))
    print(f"\n{desc} saved to memory")


def save_to_memory_file(filename, content):
    os.makedirs("memory", exist_ok=True)
    with open(f"memory/{filename}", "w", encoding="utf-8") as f:
        f.write(content)

