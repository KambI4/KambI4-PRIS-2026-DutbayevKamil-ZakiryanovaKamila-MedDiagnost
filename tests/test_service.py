from src.services.diagnostics_service import DiagnosticsService


def test_service_prioritizes_migraine_for_headache_query():
    service = DiagnosticsService()
    result = service.analyze_text("голова болит и тошнит")
    assert result["top_conditions"][0]["name"] == "Мигрень"


def test_service_detects_sore_throat_and_cold():
    service = DiagnosticsService()
    result = service.analyze_text("болит горло")
    assert "Боль в горле" in result["recognized_symptoms"]
    assert result["top_conditions"][0]["name"] == "Простуда"


def test_service_returns_expert_profile_for_lead_condition():
    service = DiagnosticsService()
    result = service.analyze_text("болит ухо и температура")
    assert result["top_conditions"][0]["name"] == "Отит"
    assert result["lead_profile"]["urgency"] in {"soon", "urgent", "plan", "watch"}
    assert result["lead_profile"]["advice"]
    assert result["lead_profile"]["when_to_seek_help"]


def test_service_asks_follow_up_for_ambiguous_cough():
    service = DiagnosticsService()
    result = service.analyze_text("у меня кашель")
    assert result["needs_clarification"] is True
    assert "сухой" in result["follow_up_question"].lower()


def test_service_uses_context_after_follow_up_answer():
    service = DiagnosticsService()
    result = service.analyze_text("сухой", context_query="у меня кашель")
    assert result["needs_clarification"] is False
    assert "Кашель" in result["recognized_symptoms"]
