import argparse
import math
import torch
import numpy as np


from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from model.stylegan2_generator import StyleGAN2Generator
from save_utils import save_pickle, load_pickle
from seed_utils import seed_fix
from visualize_utils import visualize_grid

seed_fix()


@torch.no_grad()
def Getwlatent(Mappingnet, sample_size, batch_size, latent_size, device):
    ''' This function returns list of w space latent'''

    latent_list = []
    Mappingnet.to(device)
    for _ in tqdm(range(math.ceil(sample_size / batch_size))):
        z = torch.randn(batch_size, latent_size).to(device)
        w = Mappingnet(z)['w'].detach().cpu()
        latent_list.append(w)

    latent_list = torch.cat(latent_list)
    return latent_list


def Getwbasis(w, p_component):
    ''' This function returns TOP k components '''

    print(f'Get TOP_{p_component} basis')
    transformer = IncrementalPCA(n_components=p_component, batch_size=128)
    pca = transformer.fit(w)
    basis = pca.components_
    return basis


@torch.no_grad()
def visualize_ganspace(G, component, basis_path):
    '''This function visualize result images'''

    img_list = []
    component = component
    r = np.arange(-3, 3, 0.8)

    Map_Net = G.mapping
    Tru_Net = G.truncation
    Syn_Net = G.synthesis

    saved_basis = load_pickle(basis_path)
    basis = torch.from_numpy(saved_basis[component]).float().to(device)

    fixed_z = torch.randn(1, 512).to(device)
    w = Map_Net(fixed_z)

    for i, x in enumerate(r):
        wp = w['w'] + x * basis  # Pushing our latent to basis direction.
        wp = Tru_Net(wp)
        img = Syn_Net(wp)
        img_list.append(img['image'].detach().cpu())
    img_list = torch.cat(img_list)
    visualize_grid(img_list)


if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    '''argument setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Generator name', default='stylegan2')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='FFHQ')
    parser.add_argument('--ckpt_path', type=str, help='ckpt path ', default='./pth/stylegan2_ffhq1024.pth')
    parser.add_argument('--w_path', type=str, help='w latent path', default='./latent/w.pickle')
    parser.add_argument('--N', type=int, help='Number of samples', default=1000000)
    parser.add_argument('--batch', type=int, help='Batch size ', default=32)
    parser.add_argument('--latent_dim', type=int, help='Dimension of latent z', default=512)
    parser.add_argument('--resolution', type=int, help='Generated resolution', default=1024)
    parser.add_argument('--p_component', type=int, help='Principal Components', default=20)
    parser.add_argument('--component_num', type=int, help='Component number to manipulate', default=5)
    parser.add_argument('--MODE', type=str, help='S : save latent w , P : incremental PCA , V: visualize result')
    args = parser.parse_args()

    args.basis_path = f'./latent/{args.model}_{args.dataset}_{args.p_component}.pickle'

    '''Load StyleGAN2 generator from Genforce'''
    G = StyleGAN2Generator(args.resolution).to(device)
    G.load_state_dict(torch.load(args.ckpt_path)['generator'])
    G.eval()

    '''Load Mapping & Truncation & Synthesis network'''
    Map_Net = G.mapping
    Tru_Net = G.truncation
    Syn_Net = G.synthesis

    if args.MODE == 'S':
        '''Get wlatent and saving pickle'''
        w = Getwlatent(Map_Net, args.N, args.batch, args.latent_dim, device)
        save_pickle(w, args.w_path)

    if args.MODE == 'P':
        '''Load wlatent and get principal compoents'''
        w = load_pickle(args.w_path)
        basis = Getwbasis(w, args.p_component)
        save_pickle(basis, args.basis_path)

    if args.MODE == 'V':
        '''Load principal component and visualize results'''
        visualize_ganspace(G, args.component_num, args.basis_path)
