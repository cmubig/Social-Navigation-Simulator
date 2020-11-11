from torch.utils.data import DataLoader

from gym_collision_avoidance.envs.policies.SOCIALGAN.socialgan.data.trajectories import TrajectoryDataset, seq_collate,  Custom_TrajectoryDataset


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,#args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader


def custom_data_loader(args, data_input):
    dset = Custom_TrajectoryDataset(
        data_input,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,#args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
