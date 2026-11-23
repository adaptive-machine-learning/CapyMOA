import numpy as np


class Aggregate:
    def __init__(self, n, mean, m2):
        self.n = n
        self._mean = mean
        self.m2 = m2

    def variance(self):
        if self.n < 2:
            return np.nan
        return self.m2 / (self.n - 1)

    def mean(self):
        return self._mean

    def std(self):
        if self.n < 2:
            return np.nan
        with np.errstate(all="ignore"):
            return np.sqrt(self.variance())


class PairwiseAggregate:
    def __init__(self, agg1: Aggregate, agg2: Aggregate):
        self.agg1 = agg1
        self.agg2 = agg2

    def n(self):
        return self.agg1.n, self.agg2.n

    def mean(self):
        return self.agg1.mean(), self.agg2.mean()

    def variance(self):
        return self.agg1.variance(), self.agg2.variance()

    def std(self):
        return self.agg1.std(), self.agg2.std()


class PairwiseVariance:
    def __init__(self, max_size: int):
        self.aggregates = []
        self.max_size = max_size

    def __len__(self):
        return len(self.aggregates)

    def update(self, value):
        if len(self.aggregates) == 0:
            aggregate = Aggregate(n=1, mean=value, m2=0)
            self.aggregates.append(aggregate)
        last_aggregate = self.aggregates[-1]
        count = last_aggregate.n + 1
        mean = last_aggregate.mean()
        delta = value - mean
        new_mean = mean + delta / count
        delta2 = value - new_mean
        m2 = last_aggregate.m2 + delta * delta2
        new_aggregate = Aggregate(n=count, mean=new_mean, m2=m2)
        self.aggregates.append(new_aggregate)
        if len(self.aggregates) > self.max_size:
            self.aggregates = self.aggregates[-self.max_size :]

    def reset(self):
        self.aggregates = []

    def pairwise_aggregate(self, index: int):
        agg1 = self.aggregates[index - 1]
        agg2 = self.aggregates[-1]

        n_ab = agg2.n
        n_a = agg1.n
        n_b = n_ab - n_a

        mean_ab = agg2.mean()
        mean_a = agg1.mean()
        mean_b = (n_ab * mean_ab - n_a * mean_a) / n_b

        delta = mean_b - mean_a
        m2_ab = agg2.m2
        m2_a = agg1.m2
        m2_b = m2_ab - m2_a - delta**2 * (n_a * n_b) / n_ab

        return PairwiseAggregate(agg1, Aggregate(n=n_b, mean=mean_b, m2=m2_b))
