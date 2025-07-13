# random_fixed_count_sampler.py
from torch.utils.data import Sampler
import torch, math
from mmengine.registry import DATA_SAMPLERS
from mmengine.dist import get_dist_info, sync_random_seed

@DATA_SAMPLERS.register_module()
class RandomFixedCountSampler(Sampler):
    """Sample exactly `subset_size` unique indices each epoch (DDP-aware).

    Args:
        dataset (Sized): full dataset
        subset_size (int): global number of samples to draw
        shuffle (bool): shuffle order inside the subset
        static_subset (bool): if True, keep the *same* subset every epoch
        seed (int | None): base RNG seed
    """
    def __init__(self, dataset, subset_size: int,
                 shuffle: bool = True,
                 static_subset: bool = False,
                 seed: int | None = None):

        if subset_size > len(dataset):
            print("WARNING! subset_size > len(dataset)! ", subset_size, " > ", len(dataset))
        self.dataset = dataset
        self.subset_size = subset_size
        self.shuffle = shuffle
        self.static_subset = static_subset

        self.rank, self.world_size = get_dist_info()
        self.epoch = 0
        self.seed = sync_random_seed() if seed is None else seed

        # how many samples this rank will return
        self.num_samples = math.ceil(subset_size / self.world_size)

        # subset indices pre-generated if requested
        if static_subset:
            self._subset_indices = self._generate_subset()

    # ---------- internal helpers ----------
    def _generate_subset(self):
        g = torch.Generator()
        g.manual_seed(self.seed)  # single seed for static subset
        idx = torch.randperm(len(self.dataset), generator=g).tolist()
        return idx[:self.subset_size]

    # ---------- Sampler API ----------
    def __iter__(self):
        # choose subset
        if self.static_subset:
            subset = self._subset_indices
        else:
            # epoch-dependent subset
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            full = torch.randperm(len(self.dataset), generator=g).tolist()
            subset = full[:self.subset_size]

        # shuffle order each epoch if desired (only order, not membership)
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch + 9999)
            subset = torch.tensor(subset)[torch.randperm(len(subset), generator=g)].tolist()

        # split across ranks
        return iter(subset[self.rank::self.world_size])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
