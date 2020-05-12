def fix_units(unit_str):

    unit_str = unit_str.strip()
    if len(unit_str.split(" ")) > 1:
        unit_str = unit_str.split(" ")[-1]
    unit_str = unit_str.replace("(", "")
    unit_str = unit_str.replace(")", "")

    return unit_str
