


def minMaxScale(s, min_value = None, max_value = None):
    if not min_value :
        min_value = min(s)
    if not max_value :
        max_value = max(s)
    s = (s-min_value)/(max_value-min_value)
    return s,min_value,max_value