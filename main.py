try:
    import open3d
except:
    pass
import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from data_handler import get_dataset
from torch.utils.data import DataLoader
from models import PointNet2Generator


def train(args):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    print(f'device: {device}')

    target_pc: torch.Tensor = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)
    if 0 < args.max_points < target_pc.shape[2]:
        indx = torch.randperm(target_pc.shape[2])
        target_pc = target_pc[:, :, indx[:args.cut_points]]

    data_loader = get_dataset(args.sampling_mode)(target_pc[0].transpose(0, 1), device, args)
    util.export_pc(target_pc[0], args.save_path / 'target.xyz',
                   color=torch.tensor([255, 0, 0]).unsqueeze(-1).expand(3, target_pc.shape[-1]))

    model = PointNet2Generator(device, args)


    print(f'number of parameters: {util.n_params(model)}')
    model.initialize_params(args.init_var)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    train_loader = DataLoader(data_loader, num_workers=0,
                          batch_size=args.batch_size, shuffle=False, drop_last=False)

    for i, (d1, d2) in enumerate(train_loader):
        d1, d2 = d1.to(device), d2.to(device)
        model.train()
        optimizer.zero_grad()
        d_approx = model(d2)
        loss = chamfer_distance(d_approx, d1, mse=args.mse)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'{args.save_path.name}; iter: {i} / {int(len(data_loader) / args.batch_size)}; Loss: {util.show_truncated(loss.item(), 6)};')

        if i % args.export_interval == 0:
            util.export_pc(d_approx[0], args.save_path / f'exports/export_iter:{i}.xyz')
            util.export_pc(d1[0], args.save_path / f'targets/export_iter:{i}.xyz')
            util.export_pc(d2[0], args.save_path / f'sources/export_iter:{i}.xyz')
            torch.save(model.state_dict(), args.save_path / f'generators/model{i}.pt')


if __name__ == "__main__":
    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)
    train(args)
