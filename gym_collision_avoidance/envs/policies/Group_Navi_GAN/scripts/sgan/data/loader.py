from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate, seq_delta_collate


def data_loader(args, path, shuffle=True, min_ped=1, delta=False):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim, min_ped=min_ped)
    if delta is True:
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=seq_delta_collate)
    else:
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate)
    return dset, loader
