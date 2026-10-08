"""Track actual panel edits without replacing externally automated parameters."""
import math

def edited_controls(current,previous):
    if previous is None:
        return {}
    def changed(key,value):
        if key not in previous or value==previous[key]:
            return False
        if isinstance(value,float) and isinstance(previous[key],(int,float)):
            return not math.isclose(value,previous[key],rel_tol=1e-9,abs_tol=1e-9)
        return True
    return {key:value for key,value in current.items() if changed(key,value)}

def fresh_seed(current,choose=None):
    import random
    candidate=int(choose() if choose else random.randrange(2**31))%(2**31)
    return (candidate+1)%(2**31) if candidate==current else candidate

def prompt_seed(prompt,previous,seed,locked,automated,choose=None):
    if previous is None or prompt==previous or locked or automated:return None
    return fresh_seed(seed,choose)
