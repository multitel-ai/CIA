from typing import Dict, List

def find_model_name(name: str, l: List[Dict[str, str]]) -> str:
    for small_dict in l:
        if name in small_dict:
            return small_dict[name]
    return None