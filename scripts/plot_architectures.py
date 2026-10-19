"""Draw implemented encoders, pooling, attention and classifiers as SVG diagrams.

Feature-map stacks follow the visual convention illustrated at
https://alexlenail.me/NN-SVG/AlexNet.html (drawn locally, no external assets).
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams['svg.fonttype'] = 'none'
BLUE='#e8f1fb'; ORANGE='#fff0d8'; GRAY='#edf0f3'; GREEN='#e5f4e9'
INK='#33485b'; EDGE='#728497'


def box(ax,x,y,w,h,text,color=ORANGE,fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.12',facecolor=color,edgecolor=EDGE,lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.45,color=INK)


def arrow(ax,a,b,via=None):
    if via:
        xs,ys=zip(a,*via);ax.plot(xs,ys,color=EDGE,lw=1.3);a=via[-1]
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=EDGE,lw=1.3))


def canvas(title,height=9):
    fig,ax=plt.subplots(figsize=(15,height));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.93)
    ax.set(xlim=(0,15),ylim=(0,height));ax.axis('off');fig.suptitle(title,fontsize=16,color=INK)
    return fig,ax


def text(ax,x,y,s,size=10,**kw):
    ax.text(x,y,s,fontsize=size,color=INK,linespacing=1.6,**kw)


def row(ax,nodes,y,h=1.1):
    for i,(x,w,label,*color) in enumerate(nodes):
        box(ax,x,y,w,h,label,color[0] if color else ORANGE)
        if i:arrow(ax,(nodes[i-1][0]+nodes[i-1][1],y+h/2),(x,y+h/2))


def stack(ax,x,y,size,channels,title):
    for i in reversed(range(4)):
        ax.add_patch(Rectangle((x+i*.11,y+i*.09),size,size,facecolor=BLUE,edgecolor=EDGE,lw=.9))
    for k in range(1,5):
        ax.plot([x+k*size/5]*2,[y,y+size],color='#bbcfdf',lw=.5)
        ax.plot([x,x+size],[y+k*size/5]*2,color='#bbcfdf',lw=.5)
    text(ax,x+size/2,y-.25,f'{channels} channels\n32×32 cells',ha='center',va='top',size=9)
    text(ax,x+size/2,y+size+.5,title,ha='center',size=10)


def grouping(ax,x,y):
    size=1.45
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x+j*size/4,y+i*size/4),size/4,size/4,facecolor=BLUE if (i+j)%2 else GREEN,edgecolor=EDGE,lw=1))
    text(ax,x+size/2,y-.2,'32×32 patch\n4×4 groups\n8×8 cells per group',ha='center',va='top',size=9)


def save(fig,out,name):
    path=out/f'{name}.svg'
    fig.savefig(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(out/f'{name}.png',dpi=130);plt.close(fig)


def attention(ax,y):
    text(ax,.1,y+2.05,'Inside temporal attention: one date sequence becomes one vector',size=12)
    row(ax,[(.1,2.0,'24 × 128\nLayerNorm\n+ sinusoidal date code',BLUE),
            (2.65,2.5,'Query and key Linear\n128 → 4 × 32\nValues retain 128 channels'),
            (5.7,2.5,'Mean query over dates\nBatchNorm + Linear\nOne query per head'),
            (8.75,2.5,'softmax(q·k / √32)\nDropout 0.1\nWeighted sum of dates'),
            (11.8,3.0,'4 × 128 head outputs\nConcatenate → 512',GREEN)],y+.45,h=1.35)
    text(ax,.1,y,'Then Linear 512→128 + BatchNorm + ReLU → Linear 128→128 + BatchNorm + ReLU → Dropout 0.2 → LayerNorm.',size=10)


def baseline(out):
    fig,ax=canvas('PSE+TAE: pixel aggregation, temporal attention and decoder',8.5)
    row(ax,[(.1,2.0,'Native parcel\n24 dates × 10 bands\n64 sampled pixels',BLUE),
            (2.65,2.5,'Pixel Linear 10→32\nBatchNorm + ReLU\nLinear 32→64 + BN + ReLU'),
            (5.7,2.5,'Masked pixel mean + std\n64 + 64 channels\nAppend 4 geometry values'),
            (8.75,2.5,'Date Linear 132→128\nBatchNorm\n24 date embeddings'),
            (11.8,3.0,'Temporal attention\n128-dimensional vector',GREEN)],6.4,h=1.4)
    text(ax,.1,5.85,'The same sampled pixels are followed across dates. Patch baseline: all valid 32×32 cells replace the native sample.',size=10)
    attention(ax,2.9)
    row(ax,[(.1,2.1,'Parcel embedding\n128',BLUE),(2.85,2.3,'Linear 128→64\nBatchNorm + ReLU'),
            (5.8,2.3,'Linear 64→32\nBatchNorm + ReLU'),(8.75,2.3,'Linear 32→20'),
            (11.7,3.1,'20 class logits\nSoftmax for probabilities',GREEN)],.65,h=1.15)
    save(fig,out,'baseline')


def tgt(out,scratch=False):
    fig,ax=canvas('TGT: train both paths from scratch' if scratch else 'TGT: train a correction beside a frozen baseline',9.4)
    grouping(ax,.25,6.7)
    row(ax,[(2.3,2.4,'Shared pixel MLP\n10→32→64\nBatchNorm + ReLU'),
            (5.1,2.5,'For each group and date\nMasked mean + std\n16 × 24 × 128'),
            (8.0,2.8,'Temporal attention per group\nSame weights in all groups\n16 × 128'),
            (11.2,3.5,'Across valid groups\nMean + std → 256\nLayerNorm',GREEN)],6.65,h=1.55)
    arrow(ax,(1.85,7.4),(2.3,7.4))
    text(ax,.15,8.8,'Input: 24 dates × 10 bands. Patch cells are nearest-neighbor samples; multiple cells can repeat a native pixel.',size=10)
    text(ax,2.3,5.75,'Pooling: mean = sum(valid × x) / count; std = √[sum(valid × (x − mean)²) / max(count − 1, 1)].\nEmpty groups are excluded. Group membership stays fixed across dates. TAE internals are expanded in the baseline diagram.',size=10)
    row(ax,[(.1,3.4,('Trainable patch PSE+TAE' if scratch else 'Frozen native PSE+TAE')+'\n128-dimensional embedding',ORANGE if scratch else GRAY),
            (4.0,2.7,'Concatenate\n128 + 256 = 384',BLUE),
            (7.2,3.3,'Linear 384→128\n'+('BatchNorm + ReLU' if scratch else 'ReLU + Dropout 0.2')),
            (11.0,3.7,'Linear 128→64\n'+('BatchNorm + ReLU' if scratch else 'ReLU'))],3.85,h=1.25)
    arrow(ax,(13.0,6.65),(5.35,5.1),via=[(13.0,5.4),(5.35,5.4)])
    row(ax,[(7.2,3.3,'Linear 64→20\n'+('Learned class logits' if scratch else 'Weights + bias start at zero')),
            (11.0,3.7,'20 class logits' if scratch else 'Add frozen baseline logits\n20 corrected class logits',GREEN)],1.7,h=1.15)
    arrow(ax,(12.85,3.85),(8.85,2.85),via=[(12.85,3.3),(8.85,3.3)])
    if not scratch:
        arrow(ax,(1.8,3.85),(12.85,1.7),via=[(1.8,1.25),(12.85,1.25)])
        text(ax,3,1.35,'Frozen baseline decoder output',size=9)
    text(ax,.15,.5,'Spatial-attention variant: insert a 128-wide Transformer block over the 16 groups before group pooling; no coordinates.\n4 attention heads, feed-forward width 256, dropout 0.2. Early small temporal models used width 64 rather than 128.',size=10)
    save(fig,out,'tgt-scratch' if scratch else 'tgt-correction')


def cnn(out):
    fig,ax=canvas('CNN → TGT: local feature maps, group trajectories and residual correction',11.2)
    stack(ax,.4,8.5,1.4,10,'One acquisition date')
    box(ax,2.55,8.65,2.2,1.3,'Conv 3×3\n10→32\npadding 1; stride 1')
    stack(ax,5.5,8.5,1.4,32,'Local features')
    box(ax,7.7,8.65,2.2,1.3,'Conv 3×3\n32→64\npadding 1; stride 1')
    stack(ax,10.65,8.5,1.4,64,'Local features')
    for a,b in [(2.15,2.55),(4.75,5.5),(7.25,7.7),(9.9,10.65)]:arrow(ax,(a,9.2),(b,9.2))
    ax.add_patch(Rectangle((.7,8.8),.45,.45,fill=False,edgecolor='#c85a35',lw=2))
    text(ax,13.2,9.2,'3×3 kernel\nslides over cells\n\n5×5 receptive field\nafter both layers',ha='center',va='center',size=9)
    text(ax,.2,7.35,'At each convolution: channel LayerNorm → ReLU → mask invalid cells to zero. Shared weights across all 24 dates.\nInput is band-standardized and masked. Spatial dimensions stay 32×32; no max pooling. Local context can cross group borders.',size=10)
    grouping(ax,.3,4.95)
    row(ax,[(2.5,2.7,'Within each 8×8 group\nMean + sample std\n64 + 64 channels'),
            (5.65,2.7,'Sequence per group\n24 dates × 128\n16 fixed groups',BLUE),
            (8.8,2.7,'Shared temporal attention\n4 heads; key width 32\n512→128→128'),
            (11.95,2.8,'Group mean + std\n256 → LayerNorm',GREEN)],5.1,h=1.35)
    arrow(ax,(1.9,5.8),(2.5,5.8))
    arrow(ax,(12.4,8.9),(1.0,6.4),via=[(12.6,8.9),(12.6,6.9),(1.0,6.9)])
    row(ax,[(.1,3.4,'Frozen baseline embedding\n128',GRAY),(4.0,2.7,'Concatenate\n128 + 256 = 384',BLUE),
            (7.2,3.3,'Linear 384→128\nReLU + Dropout 0.2'),(11.0,3.7,'Linear 128→64 + ReLU\nLinear 64→20; starts at zero')],2.6,h=1.2)
    arrow(ax,(13.35,5.1),(5.35,3.8),via=[(13.35,4.4),(5.35,4.4)])
    row(ax,[(.1,3.4,'Frozen baseline logits',GRAY),(4.0,2.7,'Add correction',GREEN),(7.2,3.3,'20 corrected class logits',GREEN)],.7,h=.9)
    arrow(ax,(12.85,2.6),(5.35,1.6),via=[(12.85,2.1),(5.35,2.1)])
    text(ax,11.0,.8,'CNN-spatial alternative:\npool all cells at each date,\nthen mean/std across dates;\nno temporal attention.',size=9)
    save(fig,out,'cnn-tgt')


def frozen_head(out):
    fig,ax=canvas('Replacement classifier on frozen embeddings',4.2)
    row(ax,[(.1,2.4,'Normalized frozen\nembedding: 128',GRAY),(3.0,2.7,'Linear 128→64\nLayerNorm + GELU'),
            (6.2,2.7,'Append 64 zeros\n128 inputs',BLUE),(9.4,2.4,'Linear 128→64\nGELU + Dropout 0.2'),
            (12.3,2.5,'Linear 64→20\nClass logits',GREEN)],1.7,h=1.35)
    text(ax,.2,.6,'Zero inputs retain the probe\'s shared input shape. This head replaces the baseline decoder; it does not add a residual to its logits.',size=10)
    save(fig,out,'frozen-head')


def capacity(out):
    fig,ax=canvas('Capacity controls: same patch features, wider date and temporal encoders',6.3)
    text(ax,.2,5.65,'All rows: 24 dates × 10 bands × 32×32 patch; masked pooling; pixel MLP 10→32→64; mean/std + geometry = 132 inputs.',size=10)
    for y,label,width,decoder,count in [(3.9,'Patch baseline',128,'128→64→32→20',164116),
                                        (2.2,'Capacity control',204,'204→128→64→20',345744),
                                        (.5,'Large capacity',384,'384→256→128→20',1045748)]:
        row(ax,[(.1,2.8,label+'\n'+str(count)+' parameters',BLUE),(3.4,2.8,f'Date Linear 132→{width}\nBatchNorm'),
                (6.7,4.0,f'TAE: 4 heads; key width 32\n{4*width}→{width}→{width}'),
                (11.2,3.6,'Decoder\n'+decoder, GREEN)],y,h=1.15)
    save(fig,out,'capacity')


def spectral(out):
    fig,ax=canvas('Spectral summaries and frozen-embedding classifiers',6.3)
    row(ax,[(.1,2.6,'24 dates × 10 bands\nParcel mean + std\n480 values',BLUE),
            (3.2,3.0,'6 spectral indices\n144 date values\n138 adjacent slopes\n66 seasonal summaries'),
            (6.7,3.1,'Append 4 geometry values\n480 + 144 + 138 + 66 + 4\n832 spectral features',GREEN)],3.7,h=1.55)
    box(ax,10.4,3.7,4.3,1.55,'Pairwise logistic classifier\nStandardScaler → weighted sum\n832 inputs → sigmoid',GREEN)
    arrow(ax,(9.8,4.45),(10.4,4.45))
    row(ax,[(.1,2.6,'Frozen PSE+TAE\nembedding: 128',GRAY),(3.2,3.0,'Concatenate with spectral\n128 + 832 = 960',BLUE),
            (6.7,3.1,'All-class classifier\nLogistic regression or XGBoost'),
            (10.4,4.3,'20 class probabilities',GREEN)],.8,h=1.3)
    arrow(ax,(8.25,3.7),(4.7,2.1),via=[(8.25,2.8),(4.7,2.8)])
    save(fig,out,'spectral-features')


def tabular(out):
    fig,ax=canvas('Band-moment XGBoost: inputs and model settings',4.2)
    row(ax,[(.1,3.0,'24 dates × 10 bands\nMean + std: 480 features\nWith kurtosis: 720 features',BLUE),
            (3.7,3.2,'XGBoost histogram trees\nMaximum depth 6\nLearning rate 0.1'),
            (7.5,3.2,'Row and column sampling 0.8\n200 boosting rounds\nSeed 1'),
            (11.3,3.5,'20 class probabilities',GREEN)],1.7,h=1.5)
    text(ax,.2,.65,'Select the earliest round with maximum validation mIoU. Parcels with fewer than 2 native pixels receive an all-zero feature vector.',size=10)
    save(fig,out,'tabular')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('figures/architectures'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    baseline(a.output);tgt(a.output);tgt(a.output,scratch=True);cnn(a.output)
    frozen_head(a.output);capacity(a.output);spectral(a.output);tabular(a.output)


if __name__=='__main__':main()
