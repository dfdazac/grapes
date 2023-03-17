from torch.distributions import Bernoulli, constraints
import torch


def log1mexp(x):
    assert (torch.all(x >= 0))
    return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

EPS = 1e-8

class KSubsetDistribution(torch.distributions.ExponentialFamily):
    arg_constraints = {'probs': constraints.unit_interval}
    def __init__(self, probs: torch.Tensor, K: int, log_space=True):
        # See https://arxiv.org/pdf/2210.01941.pdf, in particular Algorithm 1, 2 and 5 and 6
        assert K < probs.shape[-1]
        self.probs = probs.squeeze()
        if len(self.probs.shape) == 1:
            self.probs = self.probs.unsqueeze(0)
        self._bernoulli = Bernoulli(probs=self.probs)
        self.K = K
        self.n = self.probs.shape[-1]
        self.log_space = log_space
        self.a = self._a()
        if self.log_space:
            self.log_partition = self.a[..., self.n, self.K]
            self.partition = self.log_partition.exp()
        else:
            self.partition = self.a[..., self.n, self.K]
            self.log_partition = self.partition.log()

        super().__init__()



    def _a(self):
        # Algorithm 1 from paper
        # Code adapted from https://github.com/UCLA-StarAI/SIMPLE/blob/main/tractable_dist/k-subset_distribution.ipynb
        #  for the log-space computation
        a = torch.zeros(self.probs.shape[:-1] + (self.n + 1, self.K + 2), device=self.probs.device)
        if self.log_space:
            a = a.log()
            log_probs = self.probs.log()
        a[..., 0, 1] = 0.0 if self.log_space else 1.0
        for i in range(1, self.n + 1):
            a_as_vec = a[..., i-1, 1:]
            a_j_min_1 = a[..., i-1, :-1]
            if self.log_space:
                dont_take_i = a_as_vec + log1mexp(-log_probs[..., i-1].unsqueeze(-1))
                take_i = a_j_min_1 + log_probs[..., i-1].unsqueeze(-1)

                # Ensure the logaddexp is numerically stable by preventing the inf + inf case.
                mask = torch.logical_and(torch.isinf(dont_take_i), torch.isinf(take_i))
                a[..., i, 1:][mask] = dont_take_i[mask]
                a[..., i, 1:][~mask] = torch.logaddexp(dont_take_i[~mask], take_i[~mask])
            else:
                dont_take_i = a_as_vec * (1-self.probs[..., i-1].unsqueeze(-1))
                take_i = a_j_min_1 * self.probs[..., i-1].unsqueeze(-1)
                a[..., i, 1:] = dont_take_i + take_i

        return a[..., 1:]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        unnormalized_log_prob = self._bernoulli.log_prob(value).sum(-1)
        return unnormalized_log_prob - self.log_partition

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
                p_nomin = self.a[..., i, :][iterator, j]
                if self.log_space:
                    p[j-1 < 0] = -float('inf')
                    log_bern_prob = p + self.probs[..., i - 1].log() - p_nomin
                    bern_prob = log_bern_prob.exp()
                else:
                    p[j-1 < 0] = 0.
                    bern_prob = p * self.probs[..., i - 1] / p_nomin
                zi = torch.bernoulli(bern_prob)
                j = torch.where(zi == 1, j - 1, j)
                samples.append(zi)
            samples = torch.stack(samples, dim=-1)
            return samples.squeeze()
            # return self._sample(shape, 0, self.probs.shape[-1], self.K)


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
