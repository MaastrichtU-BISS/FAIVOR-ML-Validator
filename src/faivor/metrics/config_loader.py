import yaml
from importlib import import_module
from pathlib import Path
from faivor.metrics.metric import ModelMetric

# TODO: check the function for classification and regression models

def load_metrics(yaml_filename: str) -> tuple[list[ModelMetric], list[ModelMetric], list[ModelMetric]]:
    yaml_path = Path(__file__).parent / yaml_filename
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    overall_performance = []
    discrimination = []
    calibration = []
    classification = []
    clinical_utility = []
    fairness = []
    explainability = []

    for category in ['overall_performance', 'discrimination','calibration','classification','clinical_utility', 'fairness', 'explainability']:
        for metric_config in config.get(category, []):
            # Resolve the function/class from string
            func_str = metric_config['func']
            module_path, func_name = func_str.rsplit('.', 1)
            module = import_module(module_path)
            func = getattr(module, func_name)

            metric = ModelMetric(
                function_name=metric_config['function_name'],
                regular_name=metric_config['regular_name'],
                description=metric_config['description'],
                func=func,
                is_torch=metric_config.get('is_torch', False),
                torch_kwargs=metric_config.get('torch_kwargs', {})
            )

            if category == 'overall_performance':
                overall_performance.append(metric)
            elif category == 'classification':
                classification.append(metric)
            elif category == 'clinical_utility':
                clinical_utility.append(metric)
            elif category == 'calibration':
                calibration.append(metric)
            elif category == 'fairness':
                fairness.append(metric)
            elif category == 'discrimination':
                discrimination.append(metric)
            elif category == 'explainability':
                explainability.append(metric)

    return discrimination, calibration,classification, clinical_utility, overall_performance, fairness, explainability