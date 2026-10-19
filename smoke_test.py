"""One forward/backward pass of the paper PSE+TAE configuration, on CPU or CUDA."""

import torch
from torch.nn import functional as F

from models.stclassifier import PseTae, PseLTae, get_ntrainparams
from models.tae import MultiHeadAttention, RotaryEmbedding
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from train import configure_runtime, ClassErrorMeter, AverageValueMeter


def main():
    configure_runtime({'rdm_seed': 1, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = list(range(0, 24 * 5, 5))
    model = PseTae(
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling='mean_std',
        mlp2=[132, 128],
        with_extra=True,
        extra_size=4,
        n_head=4,
        d_k=32,
        mlp3=[512, 128, 128],
        dropout=0.2,
        T=1000,
        positions=positions,
        mlp4=[128, 64, 32, 20],
    )
    n_params = get_ntrainparams(model)
    # Official code after the Feb 2020 TAE fix (fc1_k is used, FC2 is 128-d).
    # Table 1's 147,604 is the earlier count, with FC2 at 32-d.
    print('trainable parameters: {}'.format(n_params))
    if n_params != 164116:
        raise SystemExit('unexpected parameter count {}'.format(n_params))

    model = model.to(device)
    model.apply(weight_init)
    model.train()

    batch, time, pixels = 4, 24, 64
    x = torch.randn(batch, time, 10, pixels, device=device)
    mask = torch.ones(batch, time, pixels, device=device)
    extra = torch.randn(batch, time, 4, device=device)
    y = torch.randint(0, 20, (batch,), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    criterion = FocalLoss(gamma=1)
    optimizer.zero_grad()
    out = model(((x, mask), extra))
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    acc = ClassErrorMeter(accuracy=True)
    meter = AverageValueMeter()
    acc.add(out.detach(), y)
    meter.add(loss.item())
    assert out.shape == (batch, 20), out.shape
    assert torch.isfinite(loss), loss
    probs = F.softmax(out, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch, device=device), atol=1e-4)
    print('smoke ok  device={}  loss={:.4f}  acc={:.2f}'.format(device, meter.value()[0], acc.value()[0]))

    check_rope(device)
    rope_model = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128],
        with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
        dropout=0.2, T=1000, positions=positions, pos_encoding='rope', mlp4=[128, 64, 32, 20],
    )
    if get_ntrainparams(rope_model) != 164116:
        raise SystemExit('unexpected RoPE parameter count {}'.format(get_ntrainparams(rope_model)))
    rope_model = rope_model.to(device)
    rope_model.apply(weight_init)
    rope_model.train()
    out = rope_model(((x, mask), extra))
    loss = criterion(out, y)
    loss.backward()
    assert out.shape == (batch, 20), out.shape
    assert torch.isfinite(loss), loss
    print('rope smoke ok  loss={:.4f}'.format(loss.item()))

    moment_model = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std_skew_kurtosis', mlp2=[260, 128],
        with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
        dropout=0.2, T=1000, positions=positions, mlp4=[128, 64, 32, 20],
    )
    n_moment = get_ntrainparams(moment_model)
    # The only new weights are MLP2's first layer, 132 -> 260.
    if n_moment != 180500:
        raise SystemExit('unexpected moment-pooling parameter count {}'.format(n_moment))
    moment_model = moment_model.to(device)
    moment_model.apply(weight_init)
    moment_model.train()
    out = moment_model(((x, mask), extra))
    loss = criterion(out, y)
    loss.backward()
    assert out.shape == (batch, 20), out.shape
    assert torch.isfinite(loss), loss
    assert torch.isfinite(torch.cat([p.grad.flatten() for p in moment_model.parameters() if p.grad is not None])).all()
    print('moment pooling smoke ok  params={}  loss={:.4f}'.format(n_moment, loss.item()))

    kurtosis_model = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std_kurtosis', mlp2=[196, 128],
        with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
        dropout=0.2, T=1000, positions=positions, mlp4=[128, 64, 32, 20],
    )
    n_kurtosis = get_ntrainparams(kurtosis_model)
    if n_kurtosis != 172308:
        raise SystemExit('unexpected kurtosis-pooling parameter count {}'.format(n_kurtosis))
    kurtosis_model = kurtosis_model.to(device)
    kurtosis_model.apply(weight_init)
    kurtosis_model.train()
    kurtosis_model.spatial_encoder.kurtosis_gap.fill_(10.0)
    raw = kurtosis_model.spatial_encoder.embedding_kurtosis(((x, mask), extra))
    out = kurtosis_model(((x, mask), extra))
    loss = criterion(out, y)
    loss.backward()
    assert out.shape == (batch, 20), out.shape
    assert torch.isfinite(loss), loss
    assert torch.isfinite(raw).all()
    scaled = raw / kurtosis_model.spatial_encoder.kurtosis_gap
    assert torch.allclose(scaled * 10, raw, atol=1e-4)
    print('kurtosis pooling smoke ok  params={}  loss={:.4f}'.format(n_kurtosis, loss.item()))
    enc = kurtosis_model.spatial_encoder

    def mlp1_grad_norm(stop):
        enc.kurtosis_stop_grad = stop
        kurtosis_model.zero_grad()
        out = kurtosis_model(((x, mask), extra))
        out.sum().backward()
        return float(enc.mlp1[0].lin.weight.grad.detach().norm())

    open_path = mlp1_grad_norm(False)
    stopped = mlp1_grad_norm(True)
    if abs(open_path - stopped) < 1e-8:
        raise SystemExit('stop-grad did not change the MLP1 gradient')
    print('kurtosis stop-grad ok  open {:.4f}  stopped {:.4f}'.format(open_path, stopped))

    parcel_model = PseTae(
        input_dim=10, mlp1=[30, 32, 64], pooling='mean', mlp2=[68, 128],
        with_extra=True, extra_size=4, parcel_stats='mean_std',
        n_head=4, d_k=32, mlp3=[512, 128, 128], dropout=0.2, T=1000,
        positions=positions, mlp4=[128, 64, 32, 20],
    ).to(device)
    parcel_model.apply(weight_init)
    parcel_model.train()
    out = parcel_model(((x, mask), extra))
    out.sum().backward()
    if out.shape != (batch, 20):
        raise SystemExit('parcel mean/std input failed')
    print('parcel mean/std input smoke ok')

    kurt_extra = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[142, 128],
        with_extra=True, extra_size=4, raw_kurtosis_extra=True,
        n_head=4, d_k=32, mlp3=[512, 128, 128], dropout=0.2, T=1000,
        positions=positions, mlp4=[128, 64, 32, 20],
    ).to(device)
    kurt_extra.apply(weight_init)
    kurt_extra.train()
    kurt_extra.spatial_encoder.raw_kurtosis_gap.fill_(4.0)
    out = kurt_extra(((x, mask), extra))
    out.sum().backward()
    if out.shape != (batch, 20):
        raise SystemExit('raw kurtosis extra failed')
    print('raw kurtosis extra smoke ok')

    proto_model = PseTae(
        input_dim=10, mlp1=[50, 32, 64], pooling='mean', mlp2=[68, 128],
        with_extra=True, extra_size=4, parcel_stats='mean_std', prototype_distances=True,
        n_head=4, d_k=32, mlp3=[512, 128, 128], dropout=0.2, T=1000,
        positions=positions, seq_len=24, num_classes=20, mlp4=[128, 64, 32, 20],
    ).to(device)
    proto_model.apply(weight_init)
    proto_model.spatial_encoder.prototypes.normal_()
    proto_model.train()
    out = proto_model(((x, mask), extra))
    out.sum().backward()
    if out.shape != (batch, 20):
        raise SystemExit('prototype distance input failed')
    print('prototype distance smoke ok')

    paper = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128],
        with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
        dropout=0.2, T=1000, positions=positions, mlp4=[128, 64, 32, 20],
    )
    paper.apply(weight_init)
    widened = PseTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128],
        with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
        dropout=0.2, T=1000, positions=positions, mlp4=[152, 64, 32, 20],
        date_kurtosis=True,
    ).to(device)
    path = '/tmp/psetae-baseline-smoke.pth.tar'
    torch.save({'state_dict': paper.state_dict()}, path)
    from train import graft_baseline_classifier
    graft_baseline_classifier(widened, path, device)
    widened.date_kurtosis_gap.fill_(8.0)
    widened.spatial_encoder.eval()
    widened.temporal_encoder.eval()
    widened.decoder.train()
    out = widened(((x, mask), extra))
    out.sum().backward()
    encoder_grad = widened.spatial_encoder.mlp1[0].lin.weight.grad
    if out.shape != (batch, 20) or encoder_grad is not None:
        raise SystemExit('date kurtosis graft failed')
    extra_weights = widened.decoder[0].weight[:, 128:]
    if float(extra_weights.detach().abs().sum()) != 0.0:
        raise SystemExit('new kurtosis weights were not initialized at zero')
    print('date kurtosis graft smoke ok')
    check_ltae(device, positions, ((x, mask), extra), y, criterion)


def check_ltae(device, positions, sample, y, criterion):
    model = PseLTae(
        input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128],
        with_extra=True, extra_size=4, n_head=16, d_k=8, d_model=256, mlp3=[256, 128],
        dropout=0.2, T=1000, len_max_seq=24, positions=positions, mlp4=[128, 64, 32, 20],
    )
    n_params = get_ntrainparams(model)
    print('ltae trainable parameters: {}'.format(n_params))
    # Table 1 configuration: E=256, H=16, K=8, MLP 256-128. The paper calls this ~150k.
    if n_params != 143124:
        raise SystemExit('unexpected L-TAE parameter count {}'.format(n_params))
    model = model.to(device)
    model.apply(weight_init)
    model.train()
    out = model(sample)
    loss = criterion(out, y)
    loss.backward()
    if out.shape != (y.shape[0], 20) or not torch.isfinite(loss):
        raise SystemExit('L-TAE forward failed')
    if model.temporal_encoder.attention_heads.Q.grad is None:
        raise SystemExit('L-TAE master query received no gradient')
    model.eval()
    temporal = model.temporal_encoder
    temporal.return_att = True
    with torch.no_grad():
        embedding, attn = temporal(model.spatial_encoder(sample))
    temporal.return_att = False
    if embedding.shape[-1] != 128:
        raise SystemExit('L-TAE embedding width {}'.format(embedding.shape))
    if not torch.allclose(attn.sum(dim=-1), torch.ones_like(attn[:, :, 0]), atol=1e-5):
        raise SystemExit('L-TAE attention masks do not sum to one')
    print('ltae smoke ok  params={}  loss={:.4f}'.format(n_params, loss.item()))


def check_rope(device):
    attn = MultiHeadAttention(n_head=4, d_k=32, d_in=128, rope_base=10000.0).to(device)
    attn.eval()
    q = torch.randn(2, 8, 128, device=device)
    v = torch.randn(2, 8, 128, device=device)
    pos = torch.arange(8, device=device, dtype=torch.float32)
    _, weights = attn(q, q, v, pos)
    _, shifted = attn(q, q, v, pos + 17)
    if not torch.allclose(weights, shifted, atol=1e-5):
        raise SystemExit('RoPE attention changed when every date shifted by the same amount')
    spun = RotaryEmbedding(32).to(device)
    raw = torch.randn(2, 8, 4, 32, device=device)
    rotated, _ = spun(raw, raw, pos)
    if not torch.allclose(rotated.norm(dim=-1), raw.norm(dim=-1), atol=1e-4):
        raise SystemExit('RoPE failed to preserve vector norms')


if __name__ == '__main__':
    main()
