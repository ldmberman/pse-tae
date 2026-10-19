"""Plot saved training histories; never starts or resumes a training job."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('histories',nargs='+');p.add_argument('--output',default='outputs/training.svg');a=p.parse_args()
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in a.histories:
        path=Path(name);h=json.loads(path.read_text());h=[r for r in h if 'validation' in r]
        ax.plot([r['epoch'] for r in h],[100*r['validation']['miou'] for r in h],label=path.parent.name)
    ax.set(xlabel='Epoch',ylabel='Validation mIoU (%)');ax.yaxis.set_major_locator(MultipleLocator(1));ax.grid(alpha=.2);ax.legend()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out);print(out)
if __name__=='__main__':main()
