#!/usr/bin/env python3
"""
CHB-MIT 癫痫发作检测 — 一键启动脚本

用法示例：
  # 全量流程（预处理 + 训练 + 评估，所有受试者）
  python main.py

  # 只预处理
  python main.py --mode preprocess

  # 只训练指定受试者
  python main.py --mode train --subjects chb01 chb02

  # 只评估
  python main.py --mode eval --subjects chb01

  # 调整超参数
  python main.py --subjects chb01 --epochs 50 --lr 5e-4 --threshold 0.4

  # CPU 模式
  python main.py --mode train --subjects chb01 --device cpu
"""

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.data.build_dataset import process_subject
from scripts.train import train_subject
from scripts.evaluate import evaluate_subject


def parse_args():
    parser = argparse.ArgumentParser(
        description='CHB-MIT 癫痫发作检测 Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 流程控制 ──────────────────────────────────────────────────────────
    flow = parser.add_argument_group('流程控制')
    flow.add_argument(
        '--mode', choices=['all', 'preprocess', 'train', 'eval'],
        default='all',
        help='运行阶段：all=预处理+训练+评估 / preprocess / train / eval',
    )
    flow.add_argument(
        '--subjects', nargs='+', default=['all'],
        help='受试者列表（如 chb01 chb02），或 all 表示全部',
    )
    flow.add_argument('--device', default='cuda', help='cuda 或 cpu')

    # ── 训练超参数 ─────────────────────────────────────────────────────────
    train_g = parser.add_argument_group('训练超参数')
    train_g.add_argument('--epochs', type=int, default=100)
    train_g.add_argument('--batch_size', type=int, default=64)
    train_g.add_argument('--lr', type=float, default=1e-3, help='学习率')
    train_g.add_argument('--weight_decay', type=float, default=1e-4)
    train_g.add_argument('--focal_gamma', type=float, default=2.0,
                         help='Focal Loss gamma（focal_alpha 由数据自动计算）')
    train_g.add_argument('--grad_clip', type=float, default=5.0)
    train_g.add_argument('--early_stop_patience', type=int, default=15)
    train_g.add_argument('--seed', type=int, default=42)

    # ── 模型架构 ───────────────────────────────────────────────────────────
    model_g = parser.add_argument_group('模型架构')
    model_g.add_argument('--d_model', type=int, default=128,
                         help='TCN/SE 特征维度')
    model_g.add_argument('--num_tcn_blocks', type=int, default=3,
                         help='ModernTCN 堆叠块数')
    model_g.add_argument('--lstm_hidden', type=int, default=256,
                         help='BiLSTM 每方向隐层大小')
    model_g.add_argument('--lstm_layers', type=int, default=2)

    # ── 后处理 ─────────────────────────────────────────────────────────────
    post_g = parser.add_argument_group('后处理参数')
    post_g.add_argument('--maf_window', type=int, default=5,
                        help='MAF 平滑窗口（步数，1步=1s）')
    post_g.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    post_g.add_argument('--collar_sec', type=float, default=5.0,
                        help='Collar 两端扩展秒数')

    # ── 路径 ───────────────────────────────────────────────────────────────
    path_g = parser.add_argument_group('路径配置')
    path_g.add_argument('--data_raw_dir',
                        default='/root/autodl-tmp/chb-mit-scalp-eeg-database-1.0.0',
                        help='CHB-MIT 原始数据根目录')
    path_g.add_argument('--data_processed_dir', default='/root/code/data/processed',
                        help='预处理输出目录')
    path_g.add_argument('--experiments_dir', default='/root/code/experiments',
                        help='模型权重保存目录')

    return parser.parse_args()


def build_cfg(args) -> Config:
    """用命令行参数覆盖 Config 默认值。"""
    cfg = Config(device=args.device)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.focal_gamma = args.focal_gamma
    cfg.grad_clip = args.grad_clip
    cfg.early_stop_patience = args.early_stop_patience
    cfg.seed = args.seed
    cfg.d_model = args.d_model
    cfg.num_tcn_blocks = args.num_tcn_blocks
    cfg.lstm_hidden = args.lstm_hidden
    cfg.lstm_layers = args.lstm_layers
    cfg.maf_window = args.maf_window
    cfg.threshold = args.threshold
    cfg.collar_sec = args.collar_sec
    cfg.data_raw_dir = args.data_raw_dir
    cfg.data_processed_dir = args.data_processed_dir
    return cfg


def resolve_subjects(args, cfg: Config):
    """将 'all' 展开为实际受试者列表。"""
    if args.subjects == ['all']:
        if not os.path.isdir(cfg.data_raw_dir):
            sys.exit(f"[ERROR] 原始数据目录不存在: {cfg.data_raw_dir}")
        subjects = sorted(
            d for d in os.listdir(cfg.data_raw_dir)
            if os.path.isdir(os.path.join(cfg.data_raw_dir, d))
            and d not in cfg.excluded_subjects
        )
        if not subjects:
            sys.exit(f"[ERROR] 在 {cfg.data_raw_dir} 下未找到受试者目录")
        return subjects
    return args.subjects


def main():
    args = parse_args()
    cfg = build_cfg(args)
    subjects = resolve_subjects(args, cfg)
    mode = args.mode

    print(f"\n{'='*60}")
    print(f"  模式: {mode.upper()}  |  设备: {cfg.device}")
    print(f"  受试者: {subjects}")
    print(f"  原始数据: {cfg.data_raw_dir}")
    print(f"  处理输出: {cfg.data_processed_dir}")
    print(f"  实验目录: {args.experiments_dir}")
    print(f"{'='*60}\n")

    ok, failed = [], []
    seg_results = []   # 用于 segment-based 汇总表
    evt_results = []   # 用于 event-based 汇总表

    for subject in subjects:
        print(f"\n{'─'*50}")
        print(f"  受试者: {subject}")
        print(f"{'─'*50}")

        try:
            # ── 显式预处理并保存到磁盘（仅 --mode preprocess）────────────────
            if mode == 'preprocess':
                raw_dir = os.path.join(cfg.data_raw_dir, subject)
                proc_dir = os.path.join(cfg.data_processed_dir, subject)
                if not os.path.isdir(raw_dir):
                    print(f"  [SKIP] 原始数据不存在: {raw_dir}")
                    failed.append(subject)
                    continue
                print(f"  [预处理] {raw_dir} → {proc_dir}")
                success = process_subject(raw_dir, proc_dir, cfg)
                if not success:
                    print(f"  [WARN] {subject} 预处理返回 False，跳过后续步骤")
                    failed.append(subject)
                    continue

            # ── 训练（内存中预处理，不写磁盘）──────────────────────────────
            if mode in ('all', 'train'):
                raw_dir = os.path.join(cfg.data_raw_dir, subject)
                if not os.path.isdir(raw_dir):
                    print(f"  [SKIP] 原始数据不存在: {raw_dir}")
                    failed.append(subject)
                    continue
                print(f"  [训练] subject={subject}  epochs={cfg.epochs}  lr={cfg.lr}")
                train_subject(subject, cfg, experiments_dir=args.experiments_dir)

            # ── 评估 ──────────────────────────────────────────────────────
            if mode in ('all', 'eval'):
                print(f"  [评估] subject={subject}")
                result = evaluate_subject(subject, cfg, experiments_dir=args.experiments_dir)
                if result:
                    seg_results.append((subject, result['segment']))
                    evt_results.append((subject, result['event']))
                else:
                    failed.append(subject)
                    continue

            ok.append(subject)

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {subject} 发生异常: {exc}")
            traceback.print_exc()
            failed.append(subject)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  完成: {len(ok)}/{len(subjects)} 个受试者")
    if failed:
        print(f"  失败: {failed}")
    print(f"{'='*60}\n")

    # ── Segment-based 汇总表 ─────────────────────────────────────────────
    if seg_results and mode in ('all', 'eval'):
        print("=" * 80)
        print("  Segment-Based 评价指标汇总")
        print("=" * 80)
        print(f"{'Subject':<10} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>12}")
        print("-" * 46)
        for subj, seg in seg_results:
            print(f"{subj:<10} {seg['sensitivity']:12.4f} {seg['specificity']:12.4f} {seg['accuracy']:12.4f}")
        print("-" * 46)
        if seg_results:
            avg_sens = np.mean([s['sensitivity'] for _, s in seg_results])
            avg_spec = np.mean([s['specificity'] for _, s in seg_results])
            avg_acc  = np.mean([s['accuracy'] for _, s in seg_results])
            print(f"{'Mean':<10} {avg_sens:12.4f} {avg_spec:12.4f} {avg_acc:12.4f}")
        print("=" * 80 + "\n")

    # ── Event-based 汇总表 ───────────────────────────────────────────────
    if evt_results and mode in ('all', 'eval'):
        print("=" * 100)
        print("  Event-Based 评价指标汇总")
        print("=" * 100)
        hdr = (f"{'Subject':<10} {'Dur(h)':>8} {'nTest':>7} {'nSz':>6} "
               f"{'DetSens':>10} {'FDR(/h)':>8} {'Lat(s)':>8}")
        print(hdr)
        print("-" * 60)
        for subj, evt in evt_results:
            lat_str = f"{evt['latency_s']:.2f}" if not np.isnan(evt['latency_s']) else "N/A"
            print(f"{subj:<10} {evt['duration_test_h']:8.2f} {evt['number_of_testing']:7d} "
                  f"{evt['seizures_number']:6d} {evt['true_detection_sensitivity']:10.4f} "
                  f"{evt['FDR_per_h']:8.2f} {lat_str:>8}")
        print("-" * 60)
        if evt_results:
            avg_dur    = np.mean([e['duration_test_h'] for _, e in evt_results])
            avg_ntest  = np.mean([e['number_of_testing'] for _, e in evt_results])
            avg_nsz    = np.mean([e['seizures_number'] for _, e in evt_results])
            valid_sens = [e['true_detection_sensitivity'] for _, e in evt_results
                          if not np.isnan(e['true_detection_sensitivity'])]
            valid_lat  = [e['latency_s'] for _, e in evt_results
                          if not np.isnan(e['latency_s'])]
            avg_detsens = np.mean(valid_sens) if valid_sens else float('nan')
            avg_fdr     = np.mean([e['FDR_per_h'] for _, e in evt_results])
            avg_lat     = np.mean(valid_lat) if valid_lat else float('nan')
            lat_mean_str = f"{avg_lat:.2f}" if not np.isnan(avg_lat) else "N/A"
            detsens_str  = f"{avg_detsens:.4f}" if not np.isnan(avg_detsens) else "N/A"
            print(f"{'Mean':<10} {avg_dur:8.2f} {avg_ntest:7.1f} {avg_nsz:6.1f} "
                  f"{detsens_str:>10} {avg_fdr:8.2f} {lat_mean_str:>8}")
        print("=" * 100 + "\n")

        # ── 保存 CSV ───────────────────────────────────────────────────
        seg_csv = os.path.join(args.experiments_dir, 'segment_metrics.csv')
        evt_csv = os.path.join(args.experiments_dir, 'event_metrics.csv')
        os.makedirs(args.experiments_dir, exist_ok=True)

        with open(seg_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Subject', 'Sensitivity', 'Specificity', 'Accuracy'])
            for subj, seg in seg_results:
                writer.writerow([subj, f"{seg['sensitivity']:.4f}",
                                 f"{seg['specificity']:.4f}", f"{seg['accuracy']:.4f}"])
            if seg_results:
                avg_sens = np.mean([s['sensitivity'] for _, s in seg_results])
                avg_spec = np.mean([s['specificity'] for _, s in seg_results])
                avg_acc  = np.mean([s['accuracy'] for _, s in seg_results])
                writer.writerow(['Mean', f"{avg_sens:.4f}", f"{avg_spec:.4f}", f"{avg_acc:.4f}"])
        print(f"Segment 指标已保存至: {seg_csv}")

        with open(evt_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Subject', 'Duration_h', 'nTest', 'nSz',
                             'DetSens', 'FDR_per_h', 'Latency_s'])
            for subj, evt in evt_results:
                lat_str = f"{evt['latency_s']:.2f}" if not np.isnan(evt['latency_s']) else "N/A"
                dets_str = f"{evt['true_detection_sensitivity']:.4f}" if not np.isnan(evt['true_detection_sensitivity']) else "N/A"
                writer.writerow([subj, f"{evt['duration_test_h']:.2f}",
                                 evt['number_of_testing'], evt['seizures_number'],
                                 dets_str, f"{evt['FDR_per_h']:.2f}", lat_str])
            if evt_results:
                valid_sens = [e['true_detection_sensitivity'] for _, e in evt_results
                              if not np.isnan(e['true_detection_sensitivity'])]
                valid_lat  = [e['latency_s'] for _, e in evt_results
                              if not np.isnan(e['latency_s'])]
                avg_detsens = np.mean(valid_sens) if valid_sens else float('nan')
                avg_lat = np.mean(valid_lat) if valid_lat else float('nan')
                dets_m_str = f"{avg_detsens:.4f}" if not np.isnan(avg_detsens) else "N/A"
                lat_m_str = f"{avg_lat:.2f}" if not np.isnan(avg_lat) else "N/A"
                writer.writerow(['Mean',
                                 f"{np.mean([e['duration_test_h'] for _, e in evt_results]):.2f}",
                                 f"{np.mean([e['number_of_testing'] for _, e in evt_results]):.1f}",
                                 f"{np.mean([e['seizures_number'] for _, e in evt_results]):.1f}",
                                 dets_m_str,
                                 f"{np.mean([e['FDR_per_h'] for _, e in evt_results]):.2f}",
                                 lat_m_str])
        print(f"Event 指标已保存至: {evt_csv}")


if __name__ == '__main__':
    main()
