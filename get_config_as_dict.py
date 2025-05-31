def get_config_as_dict(filepath):
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Try to convert value to int or float
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string if not a number

                config[key] = value
    return config

