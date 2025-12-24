import numpy as np
import ast
import json


def membership(x, points):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 != x1 else max(y1, y2)
    return 0.0



def fuzzify(x, lv):
    result = {}               

    for term in lv:                 
        name = term["id"]           
        points = term["points"]     

        mu = membership(x, points)  
        result[name] = mu           

    return result


def aggregate(mu_input, rules, control_lv, s_vals):
    mu = np.zeros(len(s_vals))
    for inp, out in rules:
        act = mu_input.get(inp, 0)
        term = next((t for t in control_lv if t["id"] == out), None)
        if term is None or act == 0:
            continue
        mu_term = np.array([membership(s, term["points"]) for s in s_vals])
        mu = np.maximum(mu, np.minimum(act, mu_term))
    return mu


def defuzzify(s_vals, mu):
    m = mu.max()
    if m == 0:
        return 0.0
    idx = np.where(mu == m)[0]
    return (s_vals[idx[0]] + s_vals[idx[-1]]) / 2


def main(temp_json, control_json, rules_raw, T):
    temp_lv = json.loads(temp_json)["температура"]
    control_lv = json.loads(control_json)["температура"]


    rules = ast.literal_eval(rules_raw)
    alias = {
        "интенсивно": "интенсивный",
        "умеренно": "умеренный",
        "слабо": "слабый"
    }
    rules = [(i, alias.get(o, o)) for i, o in rules]


    mu_input = fuzzify(T, temp_lv)

    xs = [p[0] for t in control_lv for p in t["points"]]
    s_vals = np.linspace(min(xs), max(xs), 1001)

    mu = aggregate(mu_input, rules, control_lv, s_vals)
    return float(defuzzify(s_vals, mu))


if __name__ == "__main__":
    with open("функции-принадлежности-температуры.json", encoding="utf-8") as f:
        temp_json = f.read()
    with open("функции-принадлежности-управление.json", encoding="utf-8") as f:
        control_json = f.read()
    with open("функция-отображения.json", encoding="utf-8") as f:
        rules_raw = f.read()

    T = 19.0
    res = main(temp_json, control_json, rules_raw, T)
    print(f"Температура {T} - оптимальный нагрев = {res:.2f}")
