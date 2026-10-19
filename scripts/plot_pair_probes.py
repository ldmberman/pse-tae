"""Export the saved pairwise probe curves with Matplotlib (no training needed)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def curve(y, score):
    order = np.argsort(-score, kind='stable')
    scores = score[order]
    tp = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    return np.r_[0., tp[ends]/y.sum()], np.r_[1., tp[ends]/(ends+1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Fold_1 output folder')
    args = parser.parse_args()
    records = json.loads((args.folder/'results.json').read_text())
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), layout='constrained')
    colors = ['#333333', '#0072b2', '#009e73', '#d55e00', '#8e44ad']
    for ax, rec in zip(axes.flat, records):
        folder = args.folder/f'{rec["weak_index"]}_{rec["conf_index"]}'
        scores = np.load(folder/'scores.npz')
        models = {m['name']: m for m in rec['models']}
        choices = ['baseline', 'embedding_logistic', 'spectral_logistic', 'embedding_spectral_logistic']
        labels = ['Baseline logits', 'Frozen embedding (linear)', 'Spectral summaries (linear)', 'Combined (linear)']
        chosen = rec['selected_by_validation_ap']
        if chosen not in choices:
            choices.append(chosen)
            labels.append('Validation-selected probe')
        for name, label, color in zip(choices, labels, colors):
            ap = rec['baseline']['test']['ap'] if name == 'baseline' else models[name]['test']['ap']
            recall, precision = curve(scores['test_labels'], scores[name+'_test'])
            assert np.isclose(np.sum(np.diff(recall)*precision[1:]), ap, atol=1e-8)
            ax.step(recall, precision, where='pre', label=f'{label}: AP {100*ap:.1f}', color=color, lw=1.4)
        ax.axhline(scores['test_labels'].mean(), color='#aaa', ls='--', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision')
        ax.set_title(f'{rec["weak"]} vs {rec["confounder"]}\n'
                     f'{rec["counts"]["test"]["weak"]} positive / {rec["counts"]["test"]["confounder"]} negative test parcels', fontsize=10)
        ax.legend(fontsize=7, loc='upper right', framealpha=.85)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(records):]:
        ax.axis('off')
        ax.text(.05, .9, 'Fold 1 — held-out test results\n\n'
            'All choices fitted on training parcels.\n'
            'Hyperparameters selected on validation AP.\n'
            'Dashed line: positive-class prevalence.\n\n'
            'AP is average precision, not accuracy or IoU.\n'
            'Pairwise gains do not establish 20-class gains.\n'
            'Rare classes have substantial sampling uncertainty.', va='top', fontsize=11, linespacing=1.6)
    fig.suptitle('Do frozen embeddings and multispectral summaries separate weak classes?', fontsize=16)
    fig.savefig(args.folder/'precision_recall.png', dpi=180)
    fig.savefig(args.folder/'precision_recall.pdf')
    print(args.folder/'precision_recall.png')


if __name__ == '__main__':
    main()
