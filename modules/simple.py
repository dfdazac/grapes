import math
from typing import Tuple, List
from torch.distributions import Bernoulli, constraints
import torch


class KSubsetDistribution(torch.distributions.ExponentialFamily):
    arg_constraints = {'probs': constraints.unit_interval}
    def __init__(self, probs: torch.Tensor, K: int):
        # See https://arxiv.org/pdf/2210.01941.pdf, in particular Algorithm 1, 2 and 5 and 6
        self.probs = probs.squeeze()
        if len(self.probs.shape) == 1:
            self.probs = self.probs.unsqueeze(0)
        self._bernoulli = Bernoulli(logits=self.probs)
        self.K = K
        self.n = self.probs.shape[-1]
        self.a = self._a()
        self.partition = self.a[..., self.n, self.K]
        super().__init__()

    def _a(self):
        # Algorithm 1 from paper
        a = torch.zeros(self.probs.shape[:-1] + (self.n + 1, self.K + 1), device=self.probs.device)
        a[..., 0, 0] = 1.0
        for i in range(1, self.n + 1):
            for j in range(self.K+1):
                dont_take_i = a[..., i-1, j] * (1-self.probs[..., i-1])
                take_i = 0.
                if j > 0:
                    take_i = a[..., i-1, j-1] * self.probs[..., i-1]
                a[..., i, j] = dont_take_i + take_i

        return a

    def pr_exactly_k(self, l: int, u: int, k: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Algorithm 5 from paper
        # Note: this algorithm doesn't work, it goes into infinite recursion. I asked the authors about this.
        if l > u:
            return 0.0
        if l == u:
            assert k == 1 or k == 0
            if k == 1:
                return self.probs[..., l]
            return 1 - self.probs[..., l]
        if (l, u) in self._cache:
            return self._cache[(l, u)]
        pm = []
        fl_u = math.floor(u / 2)
        for m in range(k + 1):
            self._cache[l, fl_u] = self.pr_exactly_k(l, fl_u, m)
            self._cache[fl_u + 1, u] = self.pr_exactly_k(fl_u + 1, u, k - m)
            pm.append(self._cache[l, fl_u] * self._cache[fl_u + 1, u])
        self.pm[(l, u)] = pm

        return sum(pm)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        unnormalized_log_prob = self._bernoulli.log_prob(value).sum(-1)
        return unnormalized_log_prob - self.partition.log()

    def sample(self, sample_shape=torch.Size()):
        # TODO: This won't work for general-shaped probs...
        b = self.probs.shape[0]
        with torch.no_grad():
            samples = []
            j = torch.full((b,), self.K, dtype=torch.long, device=self.probs.device)
            iterator = torch.arange(b, device=self.probs.device)
            for i in range(self.n, 0, -1):
                # TODO: This indexing with j probably doesn't work.
                p = self.a[..., i-1, :][iterator, j-1]
                p[j-1 < 0] = 0.
                bern_prob = p * self.probs[..., i - 1] / self.a[..., i, :][iterator, j]
                zi = torch.bernoulli(bern_prob)
                j = torch.where(zi == 1, j - 1, j)
                samples.append(zi)
            samples = torch.stack(samples, dim=-1)
            return samples.squeeze()
            # return self._sample(shape, 0, self.probs.shape[-1], self.K)

    def _sample(self, sample_shape, l: int, u: int, k: int):
        # Algorithm 6 from paper. Requires algorithm 5 which doesn't work
        probs = self.pm[(l, u)].reshape(-1, k)
        m_star = torch.multinomial(probs, sample_shape.numel(), True)
        z_lower = self._sample((1,), l, math.floor(u / 2), m_star)
        z_upper = self._sample((1,), math.floor(u / 2) + 1, u, k - m_star)
        return torch.cat([z_lower, z_upper], dim=-1)


if __name__ == "__main__":
    amt_samples = 10
    p = torch.rand((amt_samples, 20))
    k = 5
    ksubset = KSubsetDistribution(p, k)
    print(ksubset.partition)

    # Test partitions with samples
    mc_attempts = 5000
    b = torch.bernoulli(p.expand(mc_attempts, amt_samples, 20))
    sumz = b.sum(dim=-1)
    mc_partitions = (sumz == k).float().mean(dim=0)
    mse = (mc_partitions-ksubset.partition)**2
    assert (mse.mean() < 1e-5).all()

    # Test sampling algorithm
    samplez = ksubset.sample()
    assert (samplez.sum(-1) == k).all()
    print(ksubset.log_prob(samplez).exp())
