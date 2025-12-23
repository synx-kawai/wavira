#!/usr/bin/env python3
"""
CSI Human Detector - Advanced Analysis
人間/扇風機/ペットを区別する改良版検知アルゴリズム
"""

import serial
import time
import numpy as np
from scipy import signal
from scipy.stats import entropy
import re

PORT = "/dev/cu.usbserial-2120"
BAUD = 115200
COLLECT_SEC = 8  # より長い収集時間（周波数分析のため）


def parse_csi(line):
    """CSI_DATA行をパース"""
    if 'CSI_DATA' not in line:
        return None
    try:
        match = re.search(r'\[([^\]]+)\]', line)
        if not match:
            return None
        vals = [int(x) for x in match.group(1).split(',')]
        parts = line.split(',')
        amps = []
        for i in range(4, len(vals) - 1, 2):
            amps.append((vals[i]**2 + vals[i+1]**2) ** 0.5)
        return {
            'rssi': int(parts[3]) if len(parts) > 3 else 0,
            'amps': amps
        }
    except:
        return None


def collect_data(duration=COLLECT_SEC):
    """CSIデータを収集"""
    print(f'データ収集中 ({duration}秒間)...')

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1)
    ser.reset_input_buffer()

    samples = []
    rssi_vals = []
    timestamps = []
    start = time.time()

    while time.time() - start < duration:
        try:
            raw = ser.read(8192)
            if raw:
                text = raw.decode('utf-8', errors='ignore')
                for line in text.split('\n'):
                    data = parse_csi(line)
                    if data and len(data['amps']) > 0:
                        samples.append(data['amps'])
                        rssi_vals.append(data['rssi'])
                        timestamps.append(time.time() - start)
        except:
            pass

    ser.close()
    return samples, rssi_vals, timestamps


def analyze_frequency(signal_data, sample_rate):
    """FFT周波数分析"""
    n = len(signal_data)
    if n < 16:
        return None, None, None

    # ハニング窓を適用
    windowed = signal_data * np.hanning(n)

    # FFT
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1/sample_rate)

    # DC成分を除外
    fft_vals[0] = 0

    # 支配的な周波数を検出
    if len(fft_vals) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[peak_idx]
        peak_power = fft_vals[peak_idx]
        total_power = np.sum(fft_vals[1:])
        peak_ratio = peak_power / total_power if total_power > 0 else 0
    else:
        dominant_freq, peak_ratio = 0, 0

    return freqs, fft_vals, dominant_freq, peak_ratio


def analyze_periodicity(signal_data):
    """自己相関による周期性分析"""
    n = len(signal_data)
    if n < 20:
        return 0, 0

    # 正規化自己相関
    sig = signal_data - np.mean(signal_data)
    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[n-1:]  # 正のラグのみ
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

    # 最初のピークを探す（DC以外）
    min_lag = 5  # 最小ラグ
    max_lag = n // 2

    if max_lag <= min_lag:
        return 0, 0

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) == 0:
        return 0, 0

    peak_idx = np.argmax(search_region) + min_lag
    periodicity_strength = autocorr[peak_idx]

    return periodicity_strength, peak_idx


def calculate_entropy(signal_data, bins=20):
    """信号のエントロピー（不規則性）を計算"""
    if len(signal_data) < 10:
        return 0

    # ヒストグラムで確率分布を推定
    hist, _ = np.histogram(signal_data, bins=bins, density=True)
    hist = hist[hist > 0]  # ゼロを除外

    return entropy(hist)


def detect_breathing(freqs, fft_vals):
    """呼吸周波数帯（0.15-0.5Hz）のパワーを検出"""
    if freqs is None:
        return 0, 0

    # 呼吸帯域: 0.15-0.5 Hz (9-30 回/分)
    breath_mask = (freqs >= 0.15) & (freqs <= 0.5)
    breath_power = np.sum(fft_vals[breath_mask])
    total_power = np.sum(fft_vals[1:])

    breath_ratio = breath_power / total_power if total_power > 0 else 0

    # 呼吸帯域のピーク周波数
    if np.any(breath_mask):
        breath_freqs = freqs[breath_mask]
        breath_vals = fft_vals[breath_mask]
        if len(breath_vals) > 0:
            peak_breath_freq = breath_freqs[np.argmax(breath_vals)]
            return breath_ratio, peak_breath_freq

    return breath_ratio, 0


def classify_motion(analysis):
    """動きの種類を分類"""
    results = {
        'human_score': 0,
        'fan_score': 0,
        'pet_score': 0,
        'reasons': []
    }

    # 1. 周期性による判定（扇風機は高周期性）
    periodicity = analysis['periodicity_strength']
    if periodicity > 0.7:
        results['fan_score'] += 3
        results['reasons'].append(f'高い周期性 ({periodicity:.2f}) → 機械的動き')
    elif periodicity > 0.4:
        results['fan_score'] += 1
        results['reasons'].append(f'中程度の周期性 ({periodicity:.2f})')
    else:
        results['human_score'] += 1
        results['pet_score'] += 1
        results['reasons'].append(f'低い周期性 ({periodicity:.2f}) → 生物的動き')

    # 2. 呼吸検出（人間の特徴）
    breath_ratio = analysis['breath_ratio']
    if breath_ratio > 0.15:
        results['human_score'] += 3
        results['reasons'].append(f'呼吸成分検出 ({breath_ratio:.1%}) → 人間')
    elif breath_ratio > 0.08:
        results['human_score'] += 1
        results['reasons'].append(f'弱い呼吸成分 ({breath_ratio:.1%})')

    # 3. 周波数ピークの鋭さ（扇風機は鋭いピーク）
    peak_ratio = analysis['peak_ratio']
    if peak_ratio > 0.4:
        results['fan_score'] += 2
        results['reasons'].append(f'鋭い周波数ピーク ({peak_ratio:.1%}) → 機械的')

    # 4. 信号強度（体のサイズ推定）
    amp_std = analysis['temporal_std']
    mean_amp = analysis['mean_amp']

    if amp_std > 3.0 and mean_amp > 15:
        results['human_score'] += 2
        results['reasons'].append(f'大きな信号変動 (std={amp_std:.1f}) → 大きな物体')
    elif amp_std > 1.5:
        results['human_score'] += 1
        results['pet_score'] += 1

    if mean_amp < 10:
        results['pet_score'] += 1
        results['reasons'].append(f'弱い信号強度 ({mean_amp:.1f}) → 小さな物体')

    # 5. エントロピー（動きの不規則性）
    ent = analysis['entropy']
    if ent > 2.5:
        results['pet_score'] += 2
        results['reasons'].append(f'高エントロピー ({ent:.2f}) → 不規則な動き')
    elif ent > 1.8:
        results['human_score'] += 1
        results['pet_score'] += 1
    elif ent < 1.2:
        results['fan_score'] += 1
        results['reasons'].append(f'低エントロピー ({ent:.2f}) → 規則的')

    # 6. 動き指数
    motion = analysis['motion_index']
    if motion > 5.0:
        results['pet_score'] += 1
        results['reasons'].append(f'激しい動き ({motion:.1f}) → 活発')

    return results


def main():
    print('=' * 60)
    print('CSI Human Detector - 高度分析版')
    print('人間 / 扇風機 / ペット を区別')
    print('=' * 60)
    print()

    # データ収集
    samples, rssi_vals, timestamps = collect_data()

    print(f'収集完了: {len(samples)} パケット')
    print()

    if len(samples) < 30:
        print('❌ サンプル数不足（最低30パケット必要）')
        return

    # 基本分析
    all_amps = np.array(samples)
    avg_per_packet = np.mean(all_amps, axis=1)

    # サンプリングレート推定
    if len(timestamps) > 1:
        sample_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
    else:
        sample_rate = 20  # デフォルト

    # 各種分析
    temporal_std = np.std(avg_per_packet)
    mean_amp = np.mean(avg_per_packet)
    motion_index = np.std(np.diff(avg_per_packet))

    # 周波数分析
    freqs, fft_vals, dominant_freq, peak_ratio = analyze_frequency(
        avg_per_packet, sample_rate
    )

    # 周期性分析
    periodicity_strength, period_lag = analyze_periodicity(avg_per_packet)

    # エントロピー
    ent = calculate_entropy(avg_per_packet)

    # 呼吸検出
    breath_ratio, breath_freq = detect_breathing(freqs, fft_vals)

    analysis = {
        'temporal_std': temporal_std,
        'mean_amp': mean_amp,
        'motion_index': motion_index,
        'dominant_freq': dominant_freq,
        'peak_ratio': peak_ratio,
        'periodicity_strength': periodicity_strength,
        'entropy': ent,
        'breath_ratio': breath_ratio,
        'breath_freq': breath_freq,
        'sample_rate': sample_rate,
    }

    # 表示
    print('📊 基本指標')
    print('-' * 50)
    print(f'パケット数:      {len(samples)}')
    print(f'サンプリング:    {sample_rate:.1f} Hz')
    print(f'平均振幅:        {mean_amp:.2f}')
    print(f'時間変動 (std):  {temporal_std:.2f}')
    print(f'動き指数:        {motion_index:.2f}')
    print(f'平均RSSI:        {np.mean(rssi_vals):.1f} dBm')
    print()

    print('📈 周波数分析')
    print('-' * 50)
    print(f'支配周波数:      {dominant_freq:.3f} Hz')
    print(f'ピーク比率:      {peak_ratio:.1%}')
    print(f'周期性強度:      {periodicity_strength:.3f}')
    if periodicity_strength > 0.3 and period_lag > 0:
        period_sec = period_lag / sample_rate
        print(f'推定周期:        {period_sec:.2f} 秒')
    print()

    print('🫁 呼吸検出 (0.15-0.5 Hz)')
    print('-' * 50)
    print(f'呼吸帯パワー比:  {breath_ratio:.1%}')
    if breath_freq > 0:
        breath_rate = breath_freq * 60
        print(f'推定呼吸周波数:  {breath_freq:.3f} Hz ({breath_rate:.0f} 回/分)')
    print()

    print('🎲 エントロピー')
    print('-' * 50)
    print(f'動きエントロピー: {ent:.3f}')
    print()

    # 分類
    results = classify_motion(analysis)

    print('🔍 分類根拠')
    print('-' * 50)
    for reason in results['reasons']:
        print(f'  • {reason}')
    print()

    print('📋 スコア')
    print('-' * 50)
    print(f'  人間:     {results["human_score"]} 点')
    print(f'  扇風機:   {results["fan_score"]} 点')
    print(f'  ペット:   {results["pet_score"]} 点')
    print()

    # 最終判定
    scores = {
        '👤 人間': results['human_score'],
        '🌀 扇風機（機械）': results['fan_score'],
        '🐕 ペット（小動物）': results['pet_score'],
    }

    max_score = max(scores.values())

    if max_score < 2:
        print('⚪ 判定: 検知なし（無人/静止状態）')
    else:
        winners = [k for k, v in scores.items() if v == max_score]
        if len(winners) == 1:
            print(f'🟢 判定: {winners[0]}')
        else:
            print(f'🟡 判定: 複数候補 - {", ".join(winners)}')
            print('   → より長いデータ収集で精度向上可能')


if __name__ == "__main__":
    main()
