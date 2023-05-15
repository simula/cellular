def replace_val_in_dict(find_val, replace_val, var):
    if hasattr(var, "items"):
        for k, v in var.items():
            if type(v) == str and find_val in v:
                var[k] = var[k].replace(find_val, replace_val)
            if isinstance(v, dict):
                replace_val_in_dict(find_val, replace_val, v)
            elif isinstance(v, list):
                for d in v:
                    replace_val_in_dict(find_val, replace_val, d)