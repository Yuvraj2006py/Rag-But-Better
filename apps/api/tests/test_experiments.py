from app.services.experiments import default_configs, assign_config


def test_assign_config_deterministic():
    configs = default_configs()
    c1 = assign_config("query", configs)
    c2 = assign_config("query", configs)
    assert c1.config_id == c2.config_id
