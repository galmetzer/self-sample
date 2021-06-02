try:
    import open3d
except:
    pass
import options as options
import util
from models import *
from data_handler import get_dataset
import os
from pathlib import Path
from tqdm import tqdm


def eval(args):

    if not os.path.exists(args.save_path):
        Path.mkdir(args.save_path, exist_ok=True, parents=True)

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    target_pc: torch.Tensor = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)
    if args.max_points > 0:
        indx = torch.randperm(target_pc.shape[2])
        target_pc = target_pc[:, :, indx[:args.cut_points]]

    data_loader = get_dataset(args.sampling_mode)(target_pc[0].transpose(0, 1), device, args)
    model = PointNet2Generator(device, args)
    model.load_state_dict(torch.load(str(args.generator)))
    print(f'number of parameters: {util.n_params(model)}')
    model.to(device)

    torch.backends.cudnn.enabled = False
    model.eval()
    recons = []
    for i in tqdm(range(args.iterations)):
        d1, d2 = data_loader[i]
        d2 = d2.to(device)
        d_approx = model(d2.unsqueeze(0))[0].detach()
        recons.append(d_approx)

    fill_recon = torch.cat(recons, dim=1)
    if args.cat_input:
        fill_recon = torch.cat([fill_recon[:3, :].cpu(), data_loader.pc.transpose(0, 1)[:3, :].cpu()], dim=-1)

    if args.downsample != -1:
        fill_recon = util.voxel_downsample(fill_recon.transpose(1, 0).cpu(), npoints=args.downsample).transpose(1, 0)

    print('exporting result')
    util.export_pc(fill_recon, args.save_path / f'{args.name}.xyz')
    print('done!')


if __name__ == "__main__":

    parser = options.get_parser('Generate point cloud from a self-sampling generator')

    parser.add_argument('--generator', type=Path, help='path to model.pt checkpoint')
    parser.add_argument('--cat-input', action='store_true')
    parser.add_argument('--downsample', type=int, default=-1)

    args = options.parse_args(parser, inference=True)
    eval(args)
