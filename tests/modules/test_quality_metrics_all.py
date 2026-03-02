from typing import Union, get_args, get_origin

from ayase.models import QualityMetrics


def _strip_optional(tp):
    origin = get_origin(tp)
    if origin is Union:
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _sample_value(tp):
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        if args and args[0] is float:
            return [1.0]
        if args and args[0] is int:
            return [1]
        if args and args[0] is str:
            return ["x"]
        return []
    if origin is dict:
        args = get_args(tp)
        if len(args) == 2:
            key_type, value_type = args
            key = 1 if key_type is int else "k"
            if value_type is float:
                value = 1.0
            elif value_type is int:
                value = 1
            elif value_type is str:
                value = "v"
            else:
                value = "v"
            return {key: value}
        return {}
    if origin is set:
        args = get_args(tp)
        if args and args[0] is int:
            return {1}
        if args and args[0] is float:
            return {1.0}
        return {"x"}
    if origin is tuple:
        return ()
    if tp is float:
        return 1.0
    if tp is int:
        return 1
    if tp is str:
        return "x"
    if tp is bool:
        return True
    return "x"


def test_quality_metrics_accepts_all_fields():
    data = {}
    for name, field in QualityMetrics.model_fields.items():
        base = _strip_optional(field.annotation)
        data[name] = _sample_value(base)
    metrics = QualityMetrics(**data)
    for name, value in data.items():
        assert getattr(metrics, name) == value
