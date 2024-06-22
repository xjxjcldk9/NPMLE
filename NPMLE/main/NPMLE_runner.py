from NPMLE.utils.NPMLE_util import proposed_run, simulate_betas
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import os

parser = ArgumentParser()


parser.add_argument('-c', '--clean', action='count', default=0,
                    help='Delete all files that do not match with current hash number in artifact directory. Specify twice will delete anyway.')
parser.add_argument('case', choices=[
                    'normal_normal', 'beta_bernoulli'], help='Specify which case you want to run.')
parser.add_argument('-pp', '--plot_prior', action='store_true',
                    help='Plot the prior distribution to assess the NPMLE performance.')

parser.add_argument('-NP',  action='store_true',
                    help='beta estimation will calculate using NPMLE')
parser.add_argument('-TPi', action='store_true',
                    help='beta estimation will calculate using True prior grid')
parser.add_argument('-TPo', action='store_true',
                    help='beta estimation will calculate using True post')

parser.add_argument('-n', dest='n', type=int,
                    help='n size, used in plot prior.')
parser.add_argument('--cache', action='store_true',
                    help='Store beta values for N. If file exists, then will not rerun.')
parser.add_argument('--ignore_cache', action='store_true',
                    help='If file exists it will still run. Useful when B set different but parameters unchanged.')
parser.add_argument('-N', nargs='+', dest='N', type=int,
                    help='Compute corresponding betas in the second stage.')
parser.add_argument('-B', dest='B', type=int,
                    help='How many times to run to plot beta est.')
parser.add_argument('-pm', '--plot_MSE', action='store_true',
                    help='Plot all cache files if same hash number.')

args = parser.parse_args()

n = args.n
B = args.B
N = args.N

ROOT_DIR = Path(__file__).parents[2]
CASE_ARTIFACT_DIR = ROOT_DIR / 'artifacts' / f'{args.case}'
PARAMETERS_DIR = ROOT_DIR / 'NPMLE' / \
    'parameters' / f'{args.case}_parameters.py'

# 讀parameters檔，用hash去命名，這樣才不會串在一起
with open(PARAMETERS_DIR) as f:
    hashing = hashlib.md5(f.read().encode())

hash_num = hashing.hexdigest()[:5]


CASE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
(CASE_ARTIFACT_DIR / 'cache').mkdir(exist_ok=True)

# import parameters
if args.case == 'normal_normal':
    from NPMLE.parameters.normal_normal_parameters import *
elif args.case == 'beta_bernoulli':
    from NPMLE.parameters.beta_bernoulli_parameters import *

if args.NP:
    run_kwargs['NPMLE_beta_return'] = True
if args.TPi:
    run_kwargs['True_prior_grid_beta_return'] = True
if args.TPo:
    run_kwargs['True_post_mean_beta_return'] = True


if __name__ == '__main__':
    if args.clean:
        del_list = []
        del_name = []
        deeper = True if args.clean >= 2 else False

        def file_collector(dest):
            for file in os.listdir(dest):
                if Path(f'{dest}/{file}').is_file():
                    if deeper or file.split('_')[0] != hash_num:
                        del_list.append(f'{dest}/{file}')
                        del_name.append(file)

        file_collector(artifact)
        file_collector(cache_dir)

        print('You are going to delete:')
        print('\n'.join(del_name))
        if 'Y' == input('Press Y to confirm: '):
            for file in del_list:
                os.remove(file)

    if args.plot_prior:
        if not args.NP:
            raise ValueError('Must Enable NP')
        run_kwargs['generator_kwargs']['n'] = n
        proposed_run(plot=True,
                     plot_save=f'{artifact}/{hash_num}_heat_prior_{n:_}',
                     **run_kwargs)

    # 先存著
    if args.cache:
        for n in N:
            FILE_NAME = f'{cache_dir}/{hash_num}_b{B}_n{n}.csv'

            if Path(FILE_NAME).is_file() and not args.ignore_cache:
                continue
            run_kwargs['generator_kwargs']['n'] = n

            betas = simulate_betas(B, proposed_run, verbose=True,
                                   seed=20324287303, run_kwargs=run_kwargs)

            betas.to_csv(FILE_NAME, index=False)

    if args.plot_MSE:
        # 一次畫cache裡面，相同hash的東西
        CACHE_LISTS = os.listdir(cache_dir)

        beta_list = []
        for file in CACHE_LISTS:
            if file.split('_')[0] == hash_num:
                beta_list.append(pd.read_csv(f'{cache_dir}/{file}'))

        beta_dfs = pd.concat(beta_list)
        beta_dfs.groupby('case', as_index=False).apply(
            make_betas_consistent_plots, beta, artifact, hash_num)
