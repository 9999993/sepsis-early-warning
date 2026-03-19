from flask import Flask, render_template_string, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime, timedelta
import base64

app = Flask(__name__)
MODEL = None
FEATURES = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','BaseExcess','HCO3','FiO2','pH','PaCO2']
FEATURE_NAMES = {'HR':'心率','O2Sat':'血氧','Temp':'体温','SBP':'收缩压','MAP':'平均压','DBP':'舒张压','Resp':'呼吸','BaseExcess':'碱剩余','HCO3':'碳酸氢根','FiO2':'吸氧浓度','pH':'pH值','PaCO2':'二氧化碳'}
FEATURE_UNITS = {'HR':'bpm','O2Sat':'%','Temp':'°C','SBP':'mmHg','MAP':'mmHg','DBP':'mmHg','Resp':'次/分'}
NORMAL_RANGES = {'HR':(60,100),'O2Sat':(95,100),'Temp':(36.5,37.5),'SBP':(90,140),'MAP':(70,110),'DBP':(60,90),'Resp':(12,20)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM = None
PATIENTS = {}

BASELINE = {'HR':82,'O2Sat':97,'Temp':36.8,'SBP':120,'MAP':80,'DBP':65,'Resp':16,'BaseExcess':-1,'HCO3':24,'FiO2':0.30,'pH':7.39,'PaCO2':39}
VAR = {'HR':15,'O2Sat':2,'Temp':0.5,'SBP':18,'MAP':12,'DBP':10,'Resp':4,'BaseExcess':3,'HCO3':3,'FiO2':0.10,'pH':0.04,'PaCO2':6}

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.a = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.a(lstm_output), dim=1)
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

def do_norm(v):
    return [(v[f] - NORM['mean'].get(f, 0)) / (NORM['std'].get(f, 1) + 1e-8) for f in FEATURES]

def generate_reason_text(prob, importance, vitals):
    """基于模型输出生成预测原因"""
    if prob is None:
        return "⏳ 数据采集中，需要12小时历史数据..."
    
    # 找出异常指标
    abnormal = []
    if importance:
        for f, imp in importance.items():
            if imp > 0.3 and f in NORMAL_RANGES:
                val = vitals[f]
                norm = NORMAL_RANGES[f]
                if val < norm[0] or val > norm[1]:
                    direction = "升高" if val > norm[1] else "降低"
                    abnormal.append(f"{FEATURE_NAMES[f]}({val:.1f}{FEATURE_UNITS.get(f,'')}){direction}")
    
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
    
    text = f"<strong style='color:{color}'>12小时预警: {level}</strong><br>"
    text += f"<strong>脓毒症可能性: {prob*100:.1f}%</strong><br><br>"
    
    if abnormal:
        text += f"<strong>异常指标:</strong> {', '.join(abnormal[:3])}<br><br>"
    
    text += f"<strong>建议:</strong> {advice}"
    return text

def compute_importance_from_attention(hist, attn_weights):
    """基于模型注意力权重计算特征重要性"""
    if attn_weights is None or len(hist) < 12:
        return None
    
    # 注意力权重反映时间步重要性
    # 结合当前值偏离程度来推导特征重要性
    latest = hist[-1]['vitals']
    importance = {}
    
    for f in FEATURES:
        val = latest[f]
        # 使用标准化后的值作为基础
        mean = NORM['mean'].get(f, 0)
        std = NORM['std'].get(f, 1)
        z_score = abs((val - mean) / (std + 1e-8))
        
        # 结合注意力权重（最近的时间步权重更高）
        time_weight = attn_weights[-1] if attn_weights else 0.5
        
        # 特征重要性 = z_score * 时间权重
        importance[f] = min(1.0, z_score * time_weight * 0.3)
    
    # 归一化
    max_val = max(importance.values()) if importance.values() else 1
    if max_val > 0:
        importance = {k: v/max_val for k, v in importance.items()}
    
    return importance

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
    offset = {f: np.random.normal(0, VAR[f]*0.3) for f in FEATURES}
    PATIENTS[pid] = {
        'scenario': s,
        'history': [],
        'offset': offset,
        'start': np.random.randint(3, 6) if s == 'sepsis' else None  # 从第3-5小时开始恶化
    }
    return pid

def gen_hour(pid):
    p = PATIENTS[pid]
    h = len(p['history'])
    v = {}
    
    for f in FEATURES:
        base = BASELINE[f] + p['offset'][f]
        noise = np.random.normal(0, VAR[f] * 0.15)
        val = base + noise
        
        if p['scenario'] == 'sepsis' and p['start'] is not None and h > p['start']:
            hours_sick = h - p['start']
            # 更渐进的恶化：使用sigmoid曲线，开始慢，中间快，后面慢
            progress = min(1.0, hours_sick / 16)  # 16小时完全恶化
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))  # S曲线
            fluc = np.random.normal(1.0, 0.2)  # 更大的随机波动
            
            if f == 'HR': val = base * (1 + 0.25 * sigmoid * fluc) + noise
            elif f == 'O2Sat': val = base * (1 - 0.05 * sigmoid * fluc) + noise
            elif f == 'Temp': val = base + 0.8 * sigmoid * fluc + noise * 0.5
            elif f == 'SBP': val = base * (1 - 0.20 * sigmoid * fluc) + noise
            elif f == 'MAP': val = base * (1 - 0.22 * sigmoid * fluc) + noise
            elif f == 'Resp': val = base * (1 + 0.40 * sigmoid * fluc) + noise
            elif f == 'pH': val = base - 0.05 * sigmoid * fluc + noise * 0.3
            elif f == 'BaseExcess': val = base - 4 * sigmoid * fluc + noise
        
        limits = {'HR':(40,160),'O2Sat':(75,100),'Temp':(35,41),'SBP':(60,180),'MAP':(40,120),'DBP':(30,100),'Resp':(8,40),'BaseExcess':(-15,15),'HCO3':(10,40),'FiO2':(0.21,1.0),'pH':(7.0,7.6),'PaCO2':(20,80)}
        val = max(limits[f][0], min(limits[f][1], val))
        v[f] = round(val, 2) if f in ['Temp','pH','FiO2'] else round(val, 1)
    
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

@app.route('/api/performance_image')
def api_performance_image():
    try:
        with open('models/performance.png', 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        return jsonify({'ok': True, 'image': img_data})
    except:
        return jsonify({'ok': False})

if __name__ == '__main__':
    print('='*50)
    print('脓毒症12小时预警系统')
    print('='*50)
    load_model()
    print('访问: http://127.0.0.1:5001/monitoring')
    print('='*50)
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
