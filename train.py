import torch
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint


class AverageValueMeter:
    """Running mean of values, matching torchnet.meter.AverageValueMeter.value()[0]."""

    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def add(self, value, n=1):
        self.sum += float(value) * n
        self.n += n

    def value(self):
        mean = self.sum / self.n if self.n else 0.0
        return (mean, 0.0)


class ClassErrorMeter:
    """Top-1 accuracy in percent, matching torchnet.meter.ClassErrorMeter(accuracy=True)."""

    def __init__(self, accuracy=True):
        self.accuracy = accuracy
        self.correct = 0
        self.n = 0

    def add(self, output, target):
        pred = output.argmax(dim=1).view(-1)
        target = target.view(-1)
        correct = (pred == target).sum().item()
        if self.accuracy:
            self.correct += correct
        else:
            self.correct += pred.numel() - correct
        self.n += pred.numel()

    def value(self):
        score = 100.0 * self.correct / self.n if self.n else 0.0
        return (score,)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def configure_runtime(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    torch.cuda.manual_seed_all(config['rdm_seed'])
    # Full FP32. L40 tensor cores otherwise run matmul in TF32, which the
    # PyTorch 1.1 / CUDA 10 setup used for the paper did not have.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')

from models.pse import masked_mean, masked_std
from models.stclassifier import PseTae, PseLTae
from models.ltae import apply_ltae_defaults
from dataset import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss, LogitAdjustedLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

# label_44class codes kept by the paper (more than 100 parcels). Model index 0..19 follows this order.
SUB_CLASSES = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
# Figure 3 names, assigned by matching these codes to the paper's class counts (largest = Meadow).
CLASS_NAMES = {
    1: 'Meadow',
    6: 'Winter Cereal',
    5: 'Summer Cereal',
    34: 'Grapevine',
    36: 'Non-Agricultural',
    14: 'Winter Rapeseed',
    18: 'Soy',
    12: 'Leguminous Fodder',
    28: 'Fruits/Legumes/Flowers',
    9: 'Cereal',
    16: 'Sunflower',
    4: 'Spring Cereal',
    31: 'Ligneous',
    19: 'Protein Crops',
    23: 'Potato',
    8: 'Sorgho/millet/moha',
    33: 'Orchard',
    3: 'Winter Durum Wheat',
    39: 'Woodlands',
    13: 'Other Fodder',
}
EXAMPLES_PER_CLASS = 3


def class_record(index):
    code = SUB_CLASSES[int(index)]
    return code, CLASS_NAMES[code]


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], list(batch[2])
    data, target = batch
    return data, target, None


def empty_examples(n_classes):
    examples = {}
    for index in range(n_classes):
        code, name = class_record(index)
        examples[name] = {'index': index, 'code': code, 'correct': [], 'incorrect': []}
    return examples


def add_example(store, true_index, parcel_id, pred_index, confidence):
    _, name = class_record(true_index)
    pred_code, pred_name = class_record(pred_index)
    slot = 'correct' if true_index == pred_index else 'incorrect'
    bucket = store[name][slot]
    if len(bucket) >= EXAMPLES_PER_CLASS:
        return
    bucket.append({
        'parcel_id': str(parcel_id),
        'predicted_code': pred_code,
        'predicted_name': pred_name,
        'confidence': round(float(confidence), 4),
    })


def freeze_kurtosis_gap(model, data_loader, device, n_parcels=512):
    """Divide each kurtosis channel by the gap between its 1st and 99th training percentiles."""
    encoder = model.spatial_encoder
    if encoder.kurtosis_gap is None:
        return
    was_training = model.training
    model.train()
    chunks = []
    seen = 0
    for batch in data_loader:
        x, y, _ = unpack_batch(batch)
        x = recursive_todevice(x, device)
        with torch.no_grad():
            chunks.append(encoder.embedding_kurtosis(x).detach().float().cpu())
        seen += int(y.shape[0])
        if seen >= n_parcels:
            break
    data = torch.cat(chunks, dim=0)
    p01 = torch.quantile(data, 0.01, dim=0)
    p99 = torch.quantile(data, 0.99, dim=0)
    gap = (p99 - p01).clamp(min=1.0)
    encoder.kurtosis_gap.copy_(gap.to(device=encoder.kurtosis_gap.device, dtype=encoder.kurtosis_gap.dtype))
    print('KURTOSIS_GAP min {:.3f} median {:.3f} max {:.3f}'.format(
        float(gap.min()), float(gap.median()), float(gap.max())))
    if not was_training:
        model.eval()


def freeze_parcel_references(model, data_loader, device):
    """Freeze raw-kurtosis scale and class prototypes from the training parcels."""
    encoder = model.spatial_encoder
    need_gap = encoder.raw_kurtosis_gap is not None
    need_proto = encoder.prototypes is not None
    if not need_gap and not need_proto:
        return
    chunks = []
    sums = counts = None
    for batch in data_loader:
        x, y, _ = unpack_batch(batch)
        x = recursive_todevice(x, device)
        with torch.no_grad():
            if need_gap:
                chunks.append(encoder.raw_kurtosis(x).detach().float().cpu())
            if need_proto:
                pixels, mask = x[0]
                batch_size, length, channels, _ = pixels.shape
                flat_pixels = pixels.reshape(batch_size * length, channels, -1)
                flat_mask = mask.reshape(batch_size * length, -1)
                mean = masked_mean(flat_pixels, flat_mask)
                std = masked_std(flat_pixels, flat_mask)
                values = torch.cat([mean, std], dim=1).view(batch_size, length, -1).float().cpu()
                if sums is None:
                    sums = torch.zeros(encoder.num_classes, length, values.shape[-1])
                    counts = torch.zeros(encoder.num_classes)
                for row, label in zip(values, y.tolist()):
                    sums[int(label)] += row
                    counts[int(label)] += 1
    if need_gap:
        data = torch.cat(chunks, dim=0)
        gap = (torch.quantile(data, 0.99, dim=0) - torch.quantile(data, 0.01, dim=0)).clamp(min=1.0)
        encoder.raw_kurtosis_gap.copy_(gap.to(device=encoder.raw_kurtosis_gap.device, dtype=encoder.raw_kurtosis_gap.dtype))
        print('RAW_KURTOSIS_GAP min {:.3f} median {:.3f} max {:.3f}'.format(
            float(gap.min()), float(gap.median()), float(gap.max())))
    if need_proto:
        empty = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()
        prototypes = sums / counts.clamp(min=1).view(-1, 1, 1)
        encoder.prototypes.copy_(prototypes.to(device=encoder.prototypes.device, dtype=encoder.prototypes.dtype))
        print('PROTOTYPES filled {} empty {}'.format(encoder.num_classes - len(empty), empty))


def graft_baseline_classifier(model, ckpt_path, device):
    """Load a paper checkpoint and widen the classifier with zero weights on the new inputs."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    spatial = {key[len('spatial_encoder.'):]: value for key, value in state.items() if key.startswith('spatial_encoder.')}
    temporal = {key[len('temporal_encoder.'):]: value for key, value in state.items() if key.startswith('temporal_encoder.')}
    model.spatial_encoder.load_state_dict(spatial)
    model.temporal_encoder.load_state_dict(temporal)
    decoder = {key[len('decoder.'):]: value for key, value in state.items()
               if key.startswith('decoder.') and key != 'decoder.0.weight'}
    missing, unexpected = model.decoder.load_state_dict(decoder, strict=False)
    weight = state['decoder.0.weight']
    with torch.no_grad():
        model.decoder[0].weight.zero_()
        model.decoder[0].weight[:, :weight.shape[1]].copy_(weight)
        model.decoder[0].bias.copy_(state['decoder.0.bias'])
    for parameter in list(model.spatial_encoder.parameters()) + list(model.temporal_encoder.parameters()):
        parameter.requires_grad = False
    print('GRAFT decoder in_features {} extra {} missing {}'.format(
        weight.shape[1], model.decoder[0].weight.shape[1] - weight.shape[1], missing))
    if unexpected:
        raise RuntimeError('Unexpected decoder tensors: {}'.format(unexpected))


def freeze_date_kurtosis_gap(model, data_loader, device, n_parcels=512):
    chunks = []
    seen = 0
    was_training = model.spatial_encoder.training
    model.spatial_encoder.eval()
    for batch in data_loader:
        x, y, _ = unpack_batch(batch)
        x = recursive_todevice(x, device)
        chunks.append(model.spatial_encoder.date_mean_kurtosis(x).detach().float().cpu().reshape(-1))
        seen += int(y.shape[0])
        if seen >= n_parcels:
            break
    data = torch.cat(chunks)
    gap = (torch.quantile(data, 0.99) - torch.quantile(data, 0.01)).clamp(min=1.0)
    model.date_kurtosis_gap.fill_(gap.to(dtype=model.date_kurtosis_gap.dtype))
    print('DATE_KURTOSIS_GAP {:.3f} median {:.3f}'.format(float(gap), float(data.median())))
    if was_training:
        model.spatial_encoder.train()


def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = ClassErrorMeter(accuracy=True)
    loss_meter = AverageValueMeter()
    y_true = []
    y_pred = []

    for i, batch in enumerate(data_loader):
        x, y, _ = unpack_batch(batch)

        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(i + 1, len(data_loader), loss_meter.value()[0],
                                                                    acc_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])}

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = ClassErrorMeter(accuracy=True)
    loss_meter = AverageValueMeter()
    examples = empty_examples(config['num_classes']) if mode == 'val' else None

    for batch in loader:
        x, y, ids = unpack_batch(batch)
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y.long())

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())

        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        if examples is not None and ids is not None:
            confidence = torch.softmax(prediction, dim=1).max(dim=1).values.cpu().numpy()
            for parcel_id, true_index, pred_index, conf in zip(ids, y.cpu().numpy(), y_p, confidence):
                add_example(examples, int(true_index), parcel_id, int(pred_index), conf)

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}

    if mode == 'val':
        return metrics, examples
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))


def get_loaders(dt, kfold, config):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)

    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    ntest = len(indices_seq[0][1])

    loader_seq = []
    for trainval, test_indices in indices_seq:
        trainval = [indices[i] for i in trainval]
        test_indices = [indices[i] for i in test_indices]

        validation_indices = trainval[-ntest:]
        train_indices = trainval[:-ntest]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        loader_kwargs = dict(batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             pin_memory=config['device'].startswith('cuda'),
                             worker_init_fn=seed_worker if config['num_workers'] > 0 else None)
        train_loader = data.DataLoader(dt, sampler=train_sampler, **loader_kwargs)
        validation_loader = data.DataLoader(dt, sampler=validation_sampler, **loader_kwargs)
        test_loader = data.DataLoader(dt, sampler=test_sampler, **loader_kwargs)

        loader_seq.append((train_loader, validation_loader, test_loader, train_indices))
    return loader_seq


def inv_sqrt_class_weights(targets, indices, n_classes):
    counts = np.bincount([int(targets[i]) for i in indices], minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights /= weights.mean()
    return counts, weights


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    for fold in range(1, config['kfold'] + 1):
        os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def save_val_examples(fold, epoch, metrics, examples, config):
    path = os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'val_examples.jsonl')
    record = {'epoch': epoch, 'val_accuracy': metrics['val_accuracy'], 'val_IoU': metrics['val_IoU'],
              'by_class': examples}
    with open(path, 'a') as handle:
        handle.write(json.dumps(to_builtin(record)) + '\n')


def checkpoint(fold, log, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'trainlog.json'), 'w') as outfile:
        json.dump(to_builtin(log), outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'test_metrics.json'), 'w') as outfile:
        json.dump(to_builtin(metrics), outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'wb'))


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    fold_oa = []
    fold_iou = []
    n_folds = config['fold_count'] or config['kfold']
    for fold in range(1, n_folds + 1):
        fold_dir = os.path.join(config['res_dir'], 'Fold_{}'.format(fold))
        cm += pkl.load(open(os.path.join(fold_dir, 'conf_mat.pkl'), 'rb'))
        with open(os.path.join(fold_dir, 'test_metrics.json')) as handle:
            metrics = json.load(handle)
        fold_oa.append(metrics['test_accuracy'])
        fold_iou.append(100.0 * metrics['test_IoU'])

    _, perf = confusion_matrix_analysis(cm)
    oa_mean = float(np.mean(fold_oa))
    iou_mean = float(np.mean(fold_iou))
    oa_std = float(np.std(fold_oa, ddof=1)) if len(fold_oa) > 1 else 0.0
    iou_std = float(np.std(fold_iou, ddof=1)) if len(fold_iou) > 1 else 0.0

    # Table 2 of Garnot et al., CVPR 2020. OA and mIoU are means over the 5 folds.
    paper = {'OA': 94.2, 'OA_std': 0.1, 'mIoU': 50.9, 'mIoU_std': 0.8}
    perf['fold_OA_mean'] = oa_mean
    perf['fold_OA_std'] = oa_std
    perf['fold_mIoU_mean'] = iou_mean
    perf['fold_mIoU_std'] = iou_std
    perf['paper_Table2'] = paper

    print('Overall performance:')
    print('Aggregated OA: {:.2f},  macro mIoU: {:.2f}'.format(100 * perf['Accuracy'], 100 * perf['MACRO_IoU']))
    print('Fold mean OA: {:.2f} ± {:.2f}    (paper {} ± {})'.format(oa_mean, oa_std, paper['OA'], paper['OA_std']))
    print('Fold mean mIoU: {:.2f} ± {:.2f}  (paper {} ± {})'.format(iou_mean, iou_std, paper['mIoU'], paper['mIoU_std']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(to_builtin(perf), indent=4))


def main(config):
    configure_runtime(config)
    prepare_output(config)

    mean_std = pkl.load(open(config['dataset_folder'] + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    if config['preload']:
        dt = PixelSetData_preloaded(config['dataset_folder'], labels='label_44class', npixel=config['npixel'],
                          sub_classes=SUB_CLASSES,
                          norm=mean_std,
                          extra_feature=extra,
                          return_id=True)
    else:
        dt = PixelSetData(config['dataset_folder'], labels='label_44class', npixel=config['npixel'],
                          sub_classes=SUB_CLASSES,
                          norm=mean_std,
                          extra_feature=extra,
                          cache=config['cache'],
                          return_id=True)
    device = torch.device(config['device'])

    loaders = get_loaders(dt, config['kfold'], config)
    n_folds = config['fold_count'] or config['kfold']
    for fold, (train_loader, val_loader, test_loader, train_indices) in enumerate(loaders):
        if fold >= n_folds or fold + 1 < config['start_fold']:
            continue
        print('Starting Fold {}'.format(fold + 1))
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

        positions = dt.date_positions if config['positions'] == 'bespoke' else None
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            positions=positions, mlp4=config['mlp4'])

        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)
        else:
            model_config.update(with_extra=False, extra_size=None)

        if config['temporal'] == 'ltae':
            model_config.update(d_model=config['d_model'])
            model = PseLTae(**model_config)
        else:
            model_config.update(pos_encoding=config['pos_encoding'], rope_base=config['rope_base'],
                                kurtosis_stop_grad=bool(config['kurtosis_stop_grad']),
                                parcel_stats=config['parcel_stats'],
                                prototype_distances=bool(config['prototype_distances']),
                                raw_kurtosis_extra=bool(config['raw_kurtosis_extra']),
                                seq_len=config['seq_len'],
                                num_classes=config['num_classes'],
                                date_kurtosis=bool(config['date_kurtosis']))
            model = PseTae(**model_config)

        print(model.param_ratio())

        model = model.to(device)
        model.apply(weight_init)
        if config.get("kurtosis_scaling", "percentile") == "percentile":
            freeze_kurtosis_gap(model, train_loader, device)
        freeze_parcel_references(model, train_loader, device)
        if config['kurtosis_stop_grad']:
            print('KURTOSIS_STOP_GRAD 1')
        if config['parcel_stats']:
            print('PARCEL_STATS {} proto {}'.format(config['parcel_stats'], int(bool(config['prototype_distances']))))
        if config['raw_kurtosis_extra']:
            print('RAW_KURTOSIS_EXTRA 1')
        if config['date_kurtosis']:
            graft_baseline_classifier(model, config['baseline_ckpt'], device)
            freeze_date_kurtosis_gap(model, train_loader, device)
            print('DATE_KURTOSIS 1')
        if config['freeze_encoder']:
            ckpt = torch.load(config['baseline_ckpt'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            for parameter in list(model.spatial_encoder.parameters()) + list(model.temporal_encoder.parameters()):
                parameter.requires_grad = False
            print('FREEZE_ENCODER epoch {}'.format(ckpt['epoch']))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        print('OPTIMIZER PARAMETERS : {}'.format(sum(parameter.numel() for parameter in trainable)))
        optimizer = torch.optim.Adam(trainable, lr=config['lr'], betas=(0.9, 0.999))
        alpha = None
        if config['inv_sqrt_weight']:
            counts, class_weights = inv_sqrt_class_weights(dt.target, train_indices, config['num_classes'])
            alpha = class_weights.tolist()
            print('INV_SQRT_WEIGHT')
            for index, (count, weight) in enumerate(zip(counts, class_weights)):
                code, name = class_record(index)
                print('WEIGHT {} {} count {:.0f} w {:.3f}'.format(code, name, count, weight))
        if config['logit_adjust_tau'] > 0:
            counts, _ = inv_sqrt_class_weights(dt.target, train_indices, config['num_classes'])
            log_prior = np.log(counts / counts.sum())
            criterion = LogitAdjustedLoss(config['logit_adjust_tau'], log_prior)
            print('LOGIT_ADJUST_TAU {:.3f}'.format(config['logit_adjust_tau']))
            for index, offset in enumerate(config['logit_adjust_tau'] * log_prior):
                code, name = class_record(index)
                print('OFFSET {} {} {:+.3f}'.format(code, name, offset))
        else:
            criterion = FocalLoss(config['gamma'], alpha=alpha)

        trainlog = {}



        best_mIoU = 0
        for epoch in range(1, config['epochs'] + 1):
            print('EPOCH {}/{}'.format(epoch, config['epochs']))

            model.train()
            if config['date_kurtosis'] or config['freeze_encoder']:
                model.spatial_encoder.eval()
                model.temporal_encoder.eval()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

            print('Validation . . . ')
            model.eval()
            val_metrics, val_examples = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f} ({:.2f}%)'.format(
                val_metrics['val_loss'], val_metrics['val_accuracy'],
                val_metrics['val_IoU'], 100 * val_metrics['val_IoU']))

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(fold + 1, trainlog, config)
            save_val_examples(fold + 1, epoch, val_metrics, val_examples, config)

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

        print('Testing best epoch . . .')
        ckpt_path = os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar')
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['state_dict'])
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f} ({:.2f}%)'.format(
            test_metrics['test_loss'], test_metrics['test_accuracy'],
            test_metrics['test_IoU'], 100 * test_metrics['test_IoU']))
        print('FOLD_TEST {} oa {:.4f} miou {:.4f}'.format(
            fold + 1, test_metrics['test_accuracy'], 100 * test_metrics['test_IoU']))
        save_results(fold + 1, test_metrics, conf_mat, config)

    overall_performance(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, sampled tensors are frozen in RAM at initialization. '
                             'This disables per-step pixel resampling, so it does not match the paper.')
    parser.set_defaults(preload=False)
    parser.add_argument('--cache', dest='cache', action='store_true',
                        help='Load raw parcel arrays into RAM, but still resample pixels and jitter every step.')
    parser.set_defaults(cache=False)

    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--inv_sqrt_weight', action='store_true',
                        help='Multiply focal loss by the inverse square root of the training class count, '
                             'rescaled so the weights average to 1.')
    parser.set_defaults(inv_sqrt_weight=False)
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str,
                        help='Pixel-embedding pooling: underscore-separated mean, std, skew, kurtosis, max, min. '
                             'Kurtosis is divided by the training gap between its 1st and 99th percentiles. '
                             'mean_std_kurtosis needs --mlp2 [196,128], and mean_std_skew_kurtosis needs '
                             '--mlp2 [260,128], with the paper MLP1 and geometric features.')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')
    parser.add_argument('--kurtosis_scaling', choices=['percentile', 'none'], default='percentile',
                        help='none reproduces the earlier unscaled embedding-moment ablation')
    parser.add_argument('--kurtosis_stop_grad', action='store_true',
                        help='Detach kurtosis before the next linear layer so its gradient does not enter MLP1.')
    parser.set_defaults(kurtosis_stop_grad=False)
    parser.add_argument('--parcel_stats', default='', choices=['', 'mean_std', 'mean_std_kurtosis'],
                        help='Concatenate these reflectance moments onto every pixel before MLP1.')
    parser.add_argument('--prototype_distances', action='store_true',
                        help='Append distances from the parcel mean/std to twenty training-set class prototypes.')
    parser.set_defaults(prototype_distances=False)
    parser.add_argument('--raw_kurtosis_extra', action='store_true',
                        help='Concatenate gap-scaled reflectance kurtosis with the geometric features.')
    parser.set_defaults(raw_kurtosis_extra=False)
    parser.add_argument('--fold_count', default=0, type=int,
                        help='If positive, train only this many folds.')
    parser.add_argument('--start_fold', default=1, type=int,
                        help='First fold to train. Earlier folds are left as they are.')
    parser.add_argument('--seq_len', default=24, type=int, help='Dates per parcel, used to align class prototypes.')
    parser.add_argument('--date_kurtosis', action='store_true',
                        help='Append per-date channel-mean MLP1 kurtosis to the temporal embedding.')
    parser.set_defaults(date_kurtosis=False)
    parser.add_argument('--baseline_ckpt', default='', type=str,
                        help='Checkpoint to load before a frozen-encoder run.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Train only the decoder. The spatial and temporal encoders stay at --baseline_ckpt.')
    parser.set_defaults(freeze_encoder=False)
    parser.add_argument('--logit_adjust_tau', default=0.0, type=float,
                        help='If positive, train with logit-adjusted cross-entropy at this tau '
                             'instead of focal loss. Prediction remains argmax of the raw logits.')

    ## Temporal encoder. tae is the CVPR model. ltae is arXiv:2007.00586, Table 1.
    parser.add_argument('--temporal', default='tae', choices=['tae', 'ltae'],
                        help="tae keeps the CVPR temporal encoder. ltae uses the lightweight follow-up")
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--d_model', default=0, type=int,
                        help='L-TAE embedding width. 0 selects the Table 1 width, 256')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--pos_encoding', default='sin', choices=['sin', 'rope'],
                        help="Temporal position: 'sin' adds the fixed sinusoid, 'rope' rotates queries and keys")
    parser.add_argument('--rope_base', default=10000.0, type=float,
                        help='RoPE frequency base. Used only when --pos_encoding rope')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=None, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 20]', type=str, help='Number of neurons in the layers of MLP4')

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))
    if config['temporal'] == 'ltae':
        apply_ltae_defaults(config)
        blocked = [name for name in (
            'kurtosis_stop_grad', 'parcel_stats', 'prototype_distances',
            'raw_kurtosis_extra', 'date_kurtosis', 'inv_sqrt_weight', 'freeze_encoder') if config[name]]
        if config['logit_adjust_tau'] > 0:
            blocked.append('logit_adjust_tau')
        if config['pos_encoding'] != 'sin':
            blocked.append('pos_encoding')
        if blocked:
            raise SystemExit('The L-TAE baseline uses the paper pixel-set encoder and focal loss. '
                             'Drop: {}'.format(', '.join(blocked)))
    if config['freeze_encoder'] and not config['baseline_ckpt']:
        raise SystemExit('A frozen encoder needs --baseline_ckpt.')

    pprint.pprint(config)
    main(config)