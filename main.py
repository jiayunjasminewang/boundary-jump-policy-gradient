import json
import logging
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from equation import get_equation
from solver import ControlSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--dump', type=str, default='True',
                        help='Whether to dump generated samples to disk.')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'cfm_collect', 'cfm_train', 'cfm_infer'],
                        help=(
                            'train         : Phase 1+2 (critic + actor, default).\n'
                            'cfm_collect   : Phase 3a — collect (phi_0, phi*) dataset.\n'
                            'cfm_train     : Phase 3b — train CFM vector field.\n'
                            'cfm_infer     : Phase 3c — demo online inference.\n'
                            'Run train first, then cfm_collect, cfm_train, cfm_infer.'
                        ))
    # CFM-specific flags
    parser.add_argument('--cfm_instances', type=int, default=500,
                        help='Number of (phi_0, phi*) pairs to collect.')
    parser.add_argument('--cfm_grad_steps', type=int, default=300,
                        help='Max gradient steps per instance in collector.')
    parser.add_argument('--cfm_epochs', type=int, default=500,
                        help='CFM training epochs.')
    parser.add_argument('--cfm_checkpoint', type=str,
                        default='models/cfm_net',
                        help='Path to save / load CFM network weights.')
    args = parser.parse_args()

    dump = (args.dump == 'True')

    with open(args.config_path) as f:
        config_raw = json.load(f)

    # ── convert nested dicts to namedtuples for dot-access ──────────────
    config = dict_to_namespace(config_raw)

    # ── logging setup ────────────────────────────────────────────────────
    log_level = getattr(logging, config.train_config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info('Config: %s  mode: %s', args.config_path, args.mode)
    logging.info('TF version: %s', tf.__version__)

    # ── set random seeds for reproducibility ─────────────────────────────
    seed = getattr(config.train_config, 'seed', 42)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()   # TF >= 2.9
    except (AttributeError, RuntimeError):
        os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')

    # ── build equation and solver ─────────────────────────────────────────
    bsde   = get_equation(config.eqn_config)
    solver = ControlSolver(config, bsde)

    # ── generate (or load) sample trajectories ────────────────────────────
    solver.gen_samples(dump=dump, load=(not dump))

    # ── dispatch by mode ──────────────────────────────────────────────────
    if args.mode == 'train':
        ckpt_v = os.environ.get('ACTOR_TUNE_CRITIC_CKPT_VALUE', '')
        ckpt_g = os.environ.get('ACTOR_TUNE_CRITIC_CKPT_GRAD',  '')
        if ckpt_v and ckpt_g:
            solver.load_critic_weights(ckpt_v, ckpt_g)
        np.random.seed(seed + 1)
        solver.train()

    elif args.mode in ('cfm_collect', 'cfm_train', 'cfm_infer'):
        _run_cfm(args, config, solver, bsde)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')


def _run_cfm(args, config, solver, bsde):
    """Phase 3: CFM collect → train → infer pipeline."""
    from flow_matching import (PhiDataCollector, FlowMatchingNet,
                                CFMTrainer, infer_phi_star_multistart,
                                make_eval_fn)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    K, J    = bsde.K, bsde.dim
    sys_dim = FlowMatchingNet.sys_dim_for(bsde)
    net     = FlowMatchingNet(K, J, sys_dim)

    # ── cfm_collect ───────────────────────────────────────────────────
    if args.mode == 'cfm_collect':
        logging.info('CFM collect: %d instances', args.cfm_instances)
        collector = PhiDataCollector(solver, bsde)
        dataset   = collector.collect(
            num_instances = args.cfm_instances,
            grad_steps    = args.cfm_grad_steps)

        # Save dataset
        np.save('data/cfm_dataset.npy', dataset)
        logging.info('Dataset saved to data/cfm_dataset.npy')

    # ── cfm_train ─────────────────────────────────────────────────────
    elif args.mode == 'cfm_train':
        dataset = np.load('data/cfm_dataset.npy', allow_pickle=True).tolist()
        logging.info('Loaded %d pairs from data/cfm_dataset.npy', len(dataset))

        trainer = CFMTrainer(net, bsde)
        l0s, l1s, sp = trainer.prepare_dataset(dataset)
        losses = trainer.train(l0s, l1s, sp, num_epochs=args.cfm_epochs)

        net.save_weights(args.cfm_checkpoint)
        np.savetxt('data/cfm_losses.txt', losses)
        logging.info('CFM net saved to %s', args.cfm_checkpoint)

    # ── cfm_infer ─────────────────────────────────────────────────────
    elif args.mode == 'cfm_infer':
        net.load_weights(args.cfm_checkpoint)
        logging.info('CFM net loaded from %s', args.cfm_checkpoint)

        sys_params_vec = FlowMatchingNet.sys_params_np(bsde)
        eval_fn        = make_eval_fn(solver, bsde)

        phi_star, cost = infer_phi_star_multistart(
            net, sys_params_vec, K, J,
            n_starts=10, num_steps=20,
            eval_fn=eval_fn)

        logging.info('Inferred phi_star (cost=%.4f):\n%s', cost, phi_star)
        np.save('data/cfm_phi_star.npy', phi_star)
        logging.info('phi_star saved to data/cfm_phi_star.npy')


def dict_to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


if __name__ == '__main__':
    main()