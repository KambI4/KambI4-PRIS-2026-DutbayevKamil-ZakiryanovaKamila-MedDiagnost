import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'rules.json')

def load_rules():
    with open(RULES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_rules(data):
    rules = load_rules()

    # HARD FILTER
    if rules["critical_rules"]["must_be_registered"] and not data["is_registered"]:
        return "⛔ Пациент не зарегистрирован"

    if data["temperature"] > rules["thresholds"]["max_temperature"]:
        return "⚠ Очень высокая температура"

    for s in data["symptoms"]:
        if s in rules["lists"]["danger_symptoms"]:
            return f"🚨 Опасный симптом: {s}"

    return "✅ Пациент прошёл первичный осмотр"
