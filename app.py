from flask import Flask, render_template_string, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM = None
PATIENTS = {}

# 12个临床相关指标
FEATURES = [
    'HR', 'Resp', 'SBP', 'Lactate', 'PCT', 'LYM',
    'WBC', 'Urine', 'HCO3', 'GCS', 'APTT', 'O2Sat'
]

FEATURE_NAMES = {
    'HR': '心率', 'Resp': '呼吸频率', 'SBP': '收缩压',
    'Lactate': '乳酸', 'PCT': '降钙素原', 'LYM': '淋巴细胞',
    'WBC': '白细胞', 'Urine': '尿量', 'HCO3': '碳酸氢盐',
    'GCS': 'GCS评分', 'APTT': 'APTT', 'O2Sat': '血氧饱和度'
}

FEATURE_UNITS = {
    'HR': 'bpm', 'Resp': '次/分', 'SBP': 'mmHg',
    'Lactate': 'mmol/L', 'PCT': 'ng/mL', 'LYM': '×10⁹/L',
    'WBC': '×10⁹/L', 'Urine': 'mL/h', 'HCO3': 'mmol/L',
    'GCS': '分', 'APTT': '秒', 'O2Sat': '%'
}

NORMAL_RANGES = {
    'HR': (60, 100), 'Resp': (12, 20), 'SBP': (90, 140),
    'Lactate': (0.5, 2.0), 'PCT': (0, 0.5), 'LYM': (1.0, 3.0),
    'WBC': (4.0, 10.0), 'Urine': (30, 100), 'HCO3': (22, 26),
    'GCS': (13, 15), 'APTT': (25, 35), 'O2Sat': (95, 100)
}

BASELINE = {
    'HR': 82, 'Resp': 16, 'SBP': 120,
    'Lactate': 1.2, 'PCT': 0.1, 'LYM': 1.8,
    'WBC': 7.5, 'Urine': 50, 'HCO3': 24,
    'GCS': 14, 'APTT': 30, 'O2Sat': 97
}

VARIABILITY = {
    'HR': 15, 'Resp': 4, 'SBP': 18,
    'Lactate': 0.5, 'PCT': 0.1, 'LYM': 0.5,
    'WBC': 2.0, 'Urine': 20, 'HCO3': 3,
    'GCS': 1, 'APTT': 5, 'O2Sat': 2
}

LIMITS = {
    'HR': (40, 180), 'Resp': (8, 40), 'SBP': (60, 200),
    'Lactate': (0.1, 15), 'PCT': (0, 50), 'LYM': (0.1, 10),
    'WBC': (1, 30), 'Urine': (0, 200), 'HCO3': (10, 40),
    'GCS': (3, 15), 'APTT': (15, 100), 'O2Sat': (70, 100)
}

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights.squeeze(-1)

class SepsisLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = Attention(hidden_dim)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x, ret_attn=False):
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attn(lstm_out)
        output = self.clf(context).squeeze(-1)
        return (output, attn_weights) if ret_attn else output

def load_model():
    global MODEL, NORM
    ckpt = torch.load('models/sepsis_model.pt', map_location=DEVICE)
    MODEL = SepsisLSTM().to(DEVICE)
    MODEL.load_state_dict(ckpt['model_state_dict'])
    MODEL.eval()
    with open('models/norm_params.json') as f:
        NORM = json.load(f)
    print('Model loaded on', DEVICE)

# ==================== qSOFA/TIME法则 ====================
def calculate_qsofa(hr, resp, sbp, gcs):
    """计算qSOFA评分"""
    score = 0
    if resp >= 22: score += 1
    if sbp <= 100: score += 1
    if gcs < 15: score += 1
    return score

def has_infection_evidence(temp, wbc, pct):
    """判断是否有感染证据"""
    temp_abnormal = temp > 38.3 or temp < 36
    wbc_abnormal = wbc > 12 or wbc < 4
    pct_elevated = pct > 0.5
    count = sum([temp_abnormal, wbc_abnormal, pct_elevated])
    return count >= 2

def has_organ_dysfunction(lactate, urine, gcs, sbp):
    """判断是否有器官功能障碍"""
    return any([lactate > 2.0, urine < 30, gcs < 13, sbp < 90])

def check_sepsis_criteria(vitals, temp):
    """
    判断是否满足脓毒症标准
    满足以下任一条件：
    1. qSOFA ≥ 2 + 感染证据
    2. 乳酸 > 2 + 器官功能障碍
    3. 体温异常 + 心率 > 90 + 呼吸 > 22
    """
    hr = vitals.get('HR', 82)
    resp = vitals.get('Resp', 16)
    sbp = vitals.get('SBP', 120)
    lactate = vitals.get('Lactate', 1.2)
    gcs = vitals.get('GCS', 14)
    urine = vitals.get('Urine', 50)
    wbc = vitals.get('WBC', 7.5)
    pct = vitals.get('PCT', 0.1)
    
    qsofa = calculate_qsofa(hr, resp, sbp, gcs)
    infection = has_infection_evidence(temp, wbc, pct)
    organ_dysfunction = has_organ_dysfunction(lactate, urine, gcs, sbp)
    temp_abnormal = temp > 38.3 or temp < 36
    
    cond1 = qsofa >= 2 and infection
    cond2 = lactate > 2.0 and organ_dysfunction
    cond3 = temp_abnormal and hr > 90 and resp > 22
    
    return cond1 or cond2 or cond3, {
        'qsofa': qsofa,
        'has_infection': infection,
        'has_organ_dysfunction': organ_dysfunction,
        'temp_abnormal': temp_abnormal,
        'cond1': cond1,
        'cond2': cond2,
        'cond3': cond3
    }

def do_norm(v):
    return [(v[f] - NORM['mean'].get(f, 0)) / (NORM['std'].get(f, 1) + 1e-8) for f in FEATURES]

def compute_importance_from_attention(hist, attn_weights):
    if attn_weights is None or len(hist) < 12:
        return None
    latest = hist[-1]['vitals']
    importance = {}
    for f in FEATURES:
        val = latest[f]
        mean = NORM['mean'].get(f, 0)
        std = NORM['std'].get(f, 1)
        z_score = abs((val - mean) / (std + 1e-8))
        time_weight = attn_weights[-1] if attn_weights else 0.5
        importance[f] = min(1.0, z_score * time_weight * 0.3)
    max_val = max(importance.values()) if importance.values() else 1
    if max_val > 0:
        importance = {k: v/max_val for k, v in importance.items()}
    return importance

def generate_reason_text(prob, importance, vitals):
    """基于模型输出和qSOFA/TIME法则生成诊断说明"""
    if prob is None:
        return "⏳ 数据采集中，需要12小时历史数据..."
    
    # 获取体温（从其他指标推算或使用默认值）
    temp = 36.8
    if 'HR' in vitals and vitals['HR'] > 100:
        temp += 0.5
    if 'Lactate' in vitals and vitals['Lactate'] > 2:
        temp += 0.3
    
    # 检查qSOFA/TIME标准
    meets_criteria, criteria_info = check_sepsis_criteria(vitals, temp)
    
    # 收集异常指标
    abnormal = []
    if importance:
        for f, imp in importance.items():
            if imp > 0.3 and f in NORMAL_RANGES:
                val = vitals.get(f, 0)
                norm = NORMAL_RANGES[f]
                if val < norm[0] or val > norm[1]:
                    direction = "升高" if val > norm[1] else "降低"
                    abnormal.append(f"{FEATURE_NAMES.get(f, f)}({val:.1f}{FEATURE_UNITS.get(f,'')}){direction}")
    
    # 根据概率和临床标准生成说明
    if prob > 0.7:
        level = "🚨 高风险"
        color = "#ef4444"
        advice = "建议立即进行血培养检查，考虑开始经验性抗生素治疗"
    elif prob > 0.4:
        level = "⚠️ 中风险"
        color = "#fbbf24"
        advice = "建议密切监测，每2小时复查生命体征"
    else:
        level = "✅ 低风险"
        color = "#10b981"
        advice = "继续常规监测"
    
    # 构建详细说明
    text = f"<strong style='color:{color}'>12小时预警: {level}</strong><br>"
    text += f"<strong>模型预测概率: {prob*100:.1f}%</strong><br><br>"
    
    # qSOFA评分
    qsofa = criteria_info['qsofa']
    text += f"<strong>qSOFA评分: {qsofa}/3</strong>"
    if qsofa >= 2:
        text += " <span style='color:#ef4444'>(≥2分，提示脓毒症风险)</span>"
    text += "<br>"
    
    # 临床标准判断
    if meets_criteria:
        text += "<strong style='color:#ef4444'>⚠️ 满足脓毒症临床标准</strong><br>"
        if criteria_info['cond1']:
            text += "• qSOFA≥2 + 感染证据<br>"
        if criteria_info['cond2']:
            text += "• 乳酸>2 + 器官功能障碍<br>"
        if criteria_info['cond3']:
            text += "• 体温异常 + 心动过速 + 呼吸急促<br>"
    else:
        text += "<strong style='color:#10b981'>✅ 未满足脓毒症临床标准</strong><br>"
    
    text += "<br>"
    
    if abnormal:
        text += f"<strong>异常指标:</strong> {', '.join(abnormal[:3])}<br><br>"
    
    text += f"<strong>建议:</strong> {advice}"
    return text

def do_predict(hist):
    if MODEL is None or len(hist) < 12:
        return None, None, None, None
    data = [do_norm(h['vitals']) for h in hist[-12:]]
    t = torch.FloatTensor([data]).to(DEVICE)
    with torch.no_grad():
        out = MODEL(t, ret_attn=True)
        prob = out[0].item()
        attn = out[1][0].cpu().numpy().tolist() if out[1] is not None else None
    importance = compute_importance_from_attention(hist, attn)
    reason = generate_reason_text(prob, importance, hist[-1]['vitals'])
    return prob, attn, importance, reason

def gen_patient(s):
    pid = 'ICU_' + datetime.now().strftime('%H%M%S')
    offset = {f: np.random.normal(0, VARIABILITY[f]*0.3) for f in FEATURES}
    PATIENTS[pid] = {
        'scenario': s,
        'history': [],
        'offset': offset,
        'start': np.random.randint(5, 12) if s == 'sepsis' else None
    }
    return pid

def gen_hour(pid):
    p = PATIENTS[pid]
    h = len(p['history'])
    v = {}
    for f in FEATURES:
        base = BASELINE[f] + p['offset'][f]
        noise = np.random.normal(0, VARIABILITY[f] * 0.15)
        val = base + noise
        if p['scenario'] == 'sepsis' and p['start'] is not None and h > p['start']:
            hours_sick = h - p['start']
            prog = min(1.0, hours_sick / 16)
            sigmoid = 1 / (1 + np.exp(-8 * (prog - 0.5)))
            fluc = np.random.normal(1.0, 0.2)
            if f == 'HR': val = base + 30 * sigmoid * fluc + noise
            elif f == 'Resp': val = base + 10 * sigmoid * fluc + noise
            elif f == 'SBP': val = base - 30 * sigmoid * fluc + noise
            elif f == 'Lactate': val = base + 4.0 * sigmoid * fluc + noise
            elif f == 'PCT': val = base + 5.0 * sigmoid * fluc + noise
            elif f == 'LYM': val = base - 0.8 * sigmoid * fluc + noise
            elif f == 'WBC': val = base + 8.0 * sigmoid * fluc + noise
            elif f == 'Urine': val = base - 30 * sigmoid * fluc + noise
            elif f == 'HCO3': val = base - 5 * sigmoid * fluc + noise
            elif f == 'GCS': val = base - 3 * sigmoid * fluc + noise
            elif f == 'APTT': val = base + 15 * sigmoid * fluc + noise
            elif f == 'O2Sat': val = base - 6 * sigmoid * fluc + noise
        val = max(LIMITS[f][0], min(LIMITS[f][1], val))
        v[f] = round(val, 2) if f in ['Lactate', 'PCT', 'HCO3'] else (int(round(val)) if f == 'GCS' else round(val, 1))
    rec = {'hour': h, 'vitals': v, 'time': (datetime.now()-timedelta(hours=h)).strftime('%m-%d %H:%M')}
    p['history'].append(rec)
    r, a, imp, reason = do_predict(p['history'])
    lvl = '采集中' if r is None else ('高风险' if r>0.7 else ('中风险' if r>0.4 else '低风险'))
    return {
        'pid': pid, 'scenario': p['scenario'], 'hour': h, 'time': rec['time'],
        'vitals': v, 'risk': round(r,4) if r else None, 'level': lvl,
        'attention': a, 'total': len(p['history']),
        'importance': imp, 'reason_text': reason
    }

@app.route('/')
def home():
    with open('templates/index.html') as f:
        return f.read()

@app.route('/monitoring')
def mon():
    with open('templates/monitoring.html') as f:
        return render_template_string(f.read())

@app.route('/api/start/<s>')
def api_start(s):
    pid = gen_patient(s)
    return jsonify({'ok': True, 'data': gen_hour(pid)})

@app.route('/api/next/<pid>')
def api_next(pid):
    if pid not in PATIENTS:
        return jsonify({'ok': False}), 404
    return jsonify({'ok': True, 'data': gen_hour(pid)})

@app.route('/api/status')
def api_status():
    return jsonify({'model': MODEL is not None, 'device': str(DEVICE), 'patients': len(PATIENTS)})

@app.route('/api/performance')
def api_performance():
    try:
        with open('models/test_results.json') as f:
            results = json.load(f)
        with open('models/data_stats.json') as f:
            stats = json.load(f)
        return jsonify({'ok': True, 'results': results, 'stats': stats})
    except:
        return jsonify({'ok': False})

if __name__ == '__main__':
    print('='*50)
    print('脓毒症12小时预警系统 - 临床指标版')
    print('='*50)
    load_model()
    print('访问: http://127.0.0.1:5001/monitoring')
    print('='*50)
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
