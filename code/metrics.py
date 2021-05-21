import torch
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Dict


class AverageMeter(object):

    def __init__(self, total=0.0, count=0):
        self.total = total
        self.count = count

    def update(self, value):
        self.total += value
        self.count += 1

    def mean(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class Metrics(object):

    def __init__(self, metrics: List[str]):
        self._metrics = {}
        for metric in metrics:
            self._metrics[metric] = 0.0
        
    def __str__(self):
        s = []
        for key, value in self._metrics.items():
            s.append(str(key) + "=" + str(value))
        return f"Metrics({', '.join(s)})"

    def __repr__(self):
        return self.__str__()


class MetricsAverageMeter(object):

    def __init__(self, metrics: Metrics):
        self.metrics = metrics
        for key in metrics._metrics.keys():
            setattr(self, key, AverageMeter())

    def update(self, metrics: Dict):
        for key, value in metrics.items():
            getattr(self, key).update(value)

    def mean(self):
        metrics = deepcopy(self.metrics)
        for key, value in metrics._metrics.items():
            metrics._metrics[key] = getattr(self, key).mean()
        return metrics

    def reset(self):
        for key, value in self.metrics._metrics.items():
            getattr(self, key).reset()
