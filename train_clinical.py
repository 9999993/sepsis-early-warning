"""
脓毒症12小时早期预警系统 - 基于临床指标和qSOFA/TIME法则
特征：12个临床相关指标
标签：基于qSOFA评分和TIME识别法则
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# ==================== 临床指标配置 ====================
TOTAL_PATIENTS = 6000
SEPSIS_RATE = 0.15
MIN_HOURS = 24
MAX_HOURS = 60
SEQUENCE_LENGTH = 12

# 12个临床相关指标
FEATURES = [
    'HR',        # 心率 (bpm)
    'Resp',      # 呼吸频率 (次/分)
    'SBP',       # 收缩压 (mmHg)
    'Lactate',   # 乳酸 (mmol/L)
    'PCT',       # 降钙素原 (ng/mL)
    'LYM',       # 淋巴细胞计数 (×10^9/L)
    'WBC',       # 白细胞计数 (×10^9/L)
    'Urine',     # 尿量 (mL/h)
    'HCO3',      # 碳酸氢盐 (mmol/L)
    'GCS',       # 意识状态 (GCS评分)
    'APTT',      # APTT (秒)
    'O2Sat'      # 血氧饱和度 (%)
]

FEATURE_NAMES = {
    'HR': '心率', 'Resp': '呼吸频率', 'SBP': '收缩压',
    'Lactate': '乳酸', 'PCT': '降钙素原', 'LYM': '淋巴细胞',
    'WBC': '白细胞', 'Urine': '尿量', 'HCO3': '碳酸氢盐',
    'GCS': 'GCS评分', 'APTT': 'APTT', 'O2Sat': '血氧饱和度'
}

FEATURE_UNITS = {
    'HR': 'bpm', 'Resp': '次/分', 'SBP': 'mmHg',
    'Lactate': 'mmol/L', 'PCT': 'ng/mL', 'LYM': '×10^9/L',
    'WBC': '×10^9/L', 'Urine': 'mL/h', 'HCO3': 'mmol/L',
    'GCS': '分', 'APTT': '秒', 'O2Sat': '%'
}

# 正常值范围
NORMAL_RANGES = {
    'HR': (60, 100),      # 心率
    'Resp': (12, 20),     # 呼吸频率
    'SBP': (90, 140),     # 收缩压
    'Lactate': (0.5, 2.0), # 乳酸
    'PCT': (0, 0.5),      # 降钙素原
    'LYM': (1.0, 3.0),    # 淋巴细胞
    'WBC': (4.0, 10.0),   # 白细胞
    'Urine': (30, 100),   # 尿量
    'HCO3': (22, 26),     # 碳酸氢盐
    'GCS': (13, 15),      # GCS评分
    'APTT': (25, 35),     # APTT
    'O2Sat': (95, 100)    # 血氧饱和度
}

# ICU患者基线值（基于MIMIC-III统计）
BASELINE = {
    'HR': 82, 'Resp': 16, 'SBP': 120,
    'Lactate': 1.2, 'PCT': 0.1, 'LYM': 1.8,
    'WBC': 7.5, 'Urine': 50, 'HCO3': 24,
    'GCS': 14, 'APTT': 30, 'O2Sat': 97
}

# 正常波动范围
VARIABILITY = {
    'HR': 15, 'Resp': 4, 'SBP': 18,
    'Lactate': 0.5, 'PCT': 0.1, 'LYM': 0.5,
    'WBC': 2.0, 'Urine': 20, 'HCO3': 3,
    'GCS': 1, 'APTT': 5, 'O2Sat': 2
}

# 指标安全范围
LIMITS = {
    'HR': (40, 180), 'Resp': (8, 40), 'SBP': (60, 200),
    'Lactate': (0.1, 15), 'PCT': (0, 50), 'LYM': (0.1, 10),
    'WBC': (1, 30), 'Urine': (0, 200), 'HCO3': (10, 40),
    'GCS': (3, 15), 'APTT': (15, 100), 'O2Sat': (70, 100)
}

print("="*70)
print("脓毒症12小时早期预警系统 - 基于临床指标")
print("="*70)

# ==================== qSOFA评分计算 ====================
def calculate_qsofa(hr, resp, sbp, gcs):
    """
    计算qSOFA评分
    - 呼吸频率 ≥ 22次/分: 1分
    - 收缩压 ≤ 100 mmHg: 1分
    - GCS < 15: 1分
    满分3分，≥2分提示脓毒症风险
    """
    score = 0
    if resp >= 22:
        score += 1
    if sbp <= 100:
        score += 1
    if gcs < 15:
        score += 1
    return score

def has_infection_evidence(temp, wbc, pct):
    """
    判断是否有感染证据
    - 体温异常 (>38.3°C 或 <36°C)
    - 白细胞异常 (>12 或 <4)
    - PCT升高 (>0.5)
    """
    temp_abnormal = temp > 38.3 or temp < 36
    wbc_abnormal = wbc > 12 or wbc < 4
    pct_elevated = pct > 0.5
    
    # 至少满足2项
    count = sum([temp_abnormal, wbc_abnormal, pct_elevated])
    return count >= 2

def has_organ_dysfunction(lactate, urine, gcs, sbp):
    """
    判断是否有器官功能障碍
    - 乳酸 > 2 mmol/L
    - 尿量 < 30 mL/h
    - GCS < 13
    - 收缩压 < 90 mmHg
    """
    lactate_high = lactate > 2.0
    urine_low = urine < 30
    gcs_low = gcs < 13
    sbp_low = sbp < 90
    
    return any([lactate_high, urine_low, gcs_low, sbp_low])

def is_sepsis(vitals, temp=37.0):
    """
    判断是否为脓毒症
    满足以下任一条件：
    1. qSOFA ≥ 2 + 感染证据
    2. 乳酸 > 2 + 器官功能障碍
    3. 体温异常 + 心率 > 90 + 呼吸 > 22
    """
    hr = vitals['HR']
    resp = vitals['Resp']
    sbp = vitals['SBP']
    lactate = vitals['Lactate']
    gcs = vitals['GCS']
    urine = vitals['Urine']
    wbc = vitals['WBC']
    pct = vitals['PCT']
    
    # 条件1: qSOFA ≥ 2 + 感染证据
    qsofa = calculate_qsofa(hr, resp, sbp, gcs)
    infection = has_infection_evidence(temp, wbc, pct)
    condition1 = qsofa >= 2 and infection
    
    # 条件2: 乳酸 > 2 + 器官功能障碍
    organ_dysfunction = has_organ_dysfunction(lactate, urine, gcs, sbp)
    condition2 = lactate > 2.0 and organ_dysfunction
    
    # 条件3: 体温异常 + 心率 > 90 + 呼吸 > 22
    temp_abnormal = temp > 38.3 or temp < 36
    condition3 = temp_abnormal and hr > 90 and resp > 22
    
    return condition1 or condition2 or condition3

# ==================== 数据生成 ====================
print("\n[1/5] 生成6000名患者数据（基于qSOFA/TIME法则）...")

def generate_patient(pid, will_sepsis):
    """生成单个患者的时序数据"""
    n_hours = np.random.randint(MIN_HOURS, MAX_HOURS + 1)
    patient_offset = {f: np.random.normal(0, VARIABILITY[f]*0.3) for f in FEATURES}
    
    # 脓毒症恶化起始时间（第8-20小时）
    onset = np.random.randint(8, 20) if will_sepsis else None
    
    records = []
    temp_base = 36.8  # 体温基线
    consecutive_sepsis = 0  # 连续满足脓毒症条件的小时数
    
    for h in range(n_hours):
        record = {'patient_id': pid, 'hour': h}
        
        # 生成体温
        temp = temp_base + np.random.normal(0, 0.3)
        if will_sepsis and onset is not None and h > onset:
            hours_sick = h - onset
            prog = min(1.0, hours_sick / 12)
            temp += 1.5 * prog + np.random.normal(0, 0.2)
        temp = max(35, min(41, temp))
        record['Temp'] = round(temp, 1)
        
        # 生成其他指标
        for f in FEATURES:
            base = BASELINE[f] + patient_offset.get(f, 0)
            noise = np.random.normal(0, VARIABILITY[f] * 0.15)
            val = base + noise
            
            # 脓毒症恶化
            if will_sepsis and onset is not None and h > onset:
                hours_sick = h - onset
                prog = min(1.0, hours_sick / 12)
                fluc = np.random.normal(1.0, 0.15)
                
                if f == 'HR': val = base + 30 * prog * fluc + noise
                elif f == 'Resp': val = base + 10 * prog * fluc + noise
                elif f == 'SBP': val = base - 30 * prog * fluc + noise
                elif f == 'Lactate': val = base + 4.0 * prog * fluc + noise
                elif f == 'PCT': val = base + 5.0 * prog * fluc + noise
                elif f == 'LYM': val = base - 0.8 * prog * fluc + noise
                elif f == 'WBC': val = base + 8.0 * prog * fluc + noise
                elif f == 'Urine': val = base - 30 * prog * fluc + noise
                elif f == 'HCO3': val = base - 5 * prog * fluc + noise
                elif f == 'GCS': val = base - 3 * prog * fluc + noise
                elif f == 'APTT': val = base + 15 * prog * fluc + noise
                elif f == 'O2Sat': val = base - 6 * prog * fluc + noise
            
            # 限制范围
            val = max(LIMITS[f][0], min(LIMITS[f][1], val))
            
            # 保留小数位
            if f in ['Lactate', 'PCT', 'HCO3']:
                record[f] = round(val, 2)
            elif f in ['GCS']:
                record[f] = int(round(val))
            else:
                record[f] = round(val, 1)
        
        # ===== 基于qSOFA/TIME法则的标签生成 =====
        # 使用is_sepsis()函数动态判断
        is_septic_now = is_sepsis(record, record['Temp'])
        
        if is_septic_now:
            consecutive_sepsis += 1
        else:
            consecutive_sepsis = 0
        
        # 连续2小时满足条件才标记为阳性（避免单点波动）
        record['label'] = 1 if consecutive_sepsis >= 2 else 0
        
        records.append(record)
    
    return records

# 生成数据
all_records = []
sepsis_count = 0
for i in range(TOTAL_PATIENTS):
    will_sepsis = np.random.random() < SEPSIS_RATE
    if will_sepsis:
        sepsis_count += 1
    all_records.extend(generate_patient(f'P{i:05d}', will_sepsis))
    if (i + 1) % 1000 == 0:
        print(f"  已生成 {i+1}/{TOTAL_PATIENTS} 患者...")

df = pd.DataFrame(all_records)
print(f"\n数据统计:")
print(f"  总患者: {TOTAL_PATIENTS}")
print(f"  脓毒症患者: {sepsis_count} ({sepsis_count/TOTAL_PATIENTS*100:.1f}%)")
print(f"  总记录: {len(df)}")
print(f"  阳性样本: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")

# 保存数据统计
data_stats = {
    'total_patients': TOTAL_PATIENTS,
    'sepsis_patients': int(sepsis_count),
    'sepsis_rate': float(SEPSIS_RATE),
    'total_records': int(len(df)),
    'positive_samples': int(df['label'].sum()),
    'positive_rate': float(df['label'].mean()),
    'features': FEATURES,
    'feature_names': FEATURE_NAMES,
    'feature_units': FEATURE_UNITS,
    'normal_ranges': NORMAL_RANGES
}

os.makedirs('models', exist_ok=True)
with open('models/data_stats.json', 'w', encoding='utf-8') as f:
    json.dump(data_stats, f, ensure_ascii=False, indent=2)

print(f"\n特征统计（阳性样本 vs 阴性样本）:")
pos_df = df[df['label'] == 1]
neg_df = df[df['label'] == 0]
for f in FEATURES[:6]:
    pos_mean = pos_df[f].mean()
    neg_mean = neg_df[f].mean()
    diff = (pos_mean - neg_mean) / neg_mean * 100
    print(f"  {FEATURE_NAMES[f]}: {pos_mean:.2f} vs {neg_mean:.2f} ({diff:+.1f}%)")

print("\n[2/5] 分割数据集...")
pids = df['patient_id'].unique()
np.random.shuffle(pids)
n = len(pids)
train_ids = pids[:int(n*0.7)]
val_ids = pids[int(n*0.7):int(n*0.85)]
test_ids = pids[int(n*0.85):]

train_df = df[df['patient_id'].isin(train_ids)]
val_df = df[df['patient_id'].isin(val_ids)]
test_df = df[df['patient_id'].isin(test_ids)]

print(f"  训练集: {len(train_ids)}患者, {len(train_df)}记录, 阳性率{train_df['label'].mean()*100:.1f}%")
print(f"  验证集: {len(val_ids)}患者, {len(val_df)}记录, 阳性率{val_df['label'].mean()*100:.1f}%")
print(f"  测试集: {len(test_ids)}患者, {len(test_df)}记录, 阳性率{test_df['label'].mean()*100:.1f}%")

# 更新data_stats
data_stats.update({
    'train_patients': len(train_ids),
    'val_patients': len(val_ids),
    'test_patients': len(test_ids),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'train_positive_rate': float(train_df['label'].mean()),
    'val_positive_rate': float(val_df['label'].mean()),
    'test_positive_rate': float(test_df['label'].mean())
})
with open('models/data_stats.json', 'w', encoding='utf-8') as f:
    json.dump(data_stats, f, ensure_ascii=False, indent=2)

print("\n[3/5] 标准化...")
norm_params = {'mean': train_df[FEATURES].mean().to_dict(), 'std': train_df[FEATURES].std().to_dict()}
with open('models/norm_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# ==================== 数据集 ====================
class SepsisDataset(Dataset):
    def __init__(self, df, norm, seq_len=12):
        self.seq_len = seq_len
        self.data = df.copy()
        for feat in FEATURES:
            self.data[feat] = (self.data[feat] - norm['mean'][feat]) / (norm['std'][feat] + 1e-8)
        
        self.seqs, self.labels = [], []
        for pid in self.data['patient_id'].unique():
            p = self.data[self.data['patient_id']==pid].sort_values('hour')
            if len(p) >= seq_len:
                vals = p[FEATURES].values
                labs = p['label'].values
                for i in range(len(p) - seq_len + 1):
                    self.seqs.append(vals[i:i+seq_len])
                    self.labels.append(labs[i+seq_len-1])
    
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return torch.FloatTensor(self.seqs[i]), torch.FloatTensor([self.labels[i]])

train_ds = SepsisDataset(train_df, norm_params)
val_ds = SepsisDataset(val_df, norm_params)
test_ds = SepsisDataset(test_df, norm_params)
print(f"  序列: 训练{len(train_ds)}, 验证{len(val_ds)}, 测试{len(test_ds)}")

# ==================== 模型定义 ====================
print("\n[4/5] 定义BiLSTM-Attention模型...")

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

model = SepsisLSTM()
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 训练 ====================
print("\n[5/5] 开始训练（30轮）...")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
criterion = nn.BCELoss()

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy().flatten())
    preds, labels = np.array(all_preds), np.array(all_labels)
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, (preds > 0.5).astype(int), zero_division=0)
    precision = precision_score(labels, (preds > 0.5).astype(int), zero_division=0)
    recall = recall_score(labels, (preds > 0.5).astype(int), zero_division=0)
    return auc, f1, precision, recall, preds, labels

history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
best_auc = 0

for epoch in range(30):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.squeeze().to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    va, vf, vp, vr, _, _ = evaluate(val_loader)
    
    history['train_loss'].append(total_loss / len(train_loader))
    history['val_auc'].append(va)
    history['val_f1'].append(vf)
    
    if va > best_auc:
        best_auc = va
        torch.save({'model_state_dict': model.state_dict(), 'val_auc': va, 'epoch': epoch}, 'models/sepsis_model.pt')
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/30: Loss={total_loss/len(train_loader):.4f}, AUC={va:.4f}, F1={vf:.4f}")

# ==================== 测试评估 ====================
print("\n" + "="*70)
print("测试集评估")
print("="*70)

ckpt = torch.load('models/sepsis_model.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
ta, tf, tp, tr, test_preds, test_labels = evaluate(test_loader)

print(f"\n最佳验证AUC: {best_auc:.4f}")
print(f"\n测试集结果:")
print(f"  AUC-ROC: {ta:.4f}")
print(f"  F1-Score: {tf:.4f}")
print(f"  Precision: {tp:.4f}")
print(f"  Recall: {tr:.4f}")

# 保存结果
results = {
    'best_epoch': ckpt['epoch'],
    'val_auc': float(best_auc),
    'test_auc': float(ta),
    'test_f1': float(tf),
    'test_precision': float(tp),
    'test_recall': float(tr)
}
with open('models/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ==================== 生成性能图 ====================
print("\n生成性能图...")

# 性能图1: ROC, PR, Loss, Metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sepsis 12-Hour Early Warning System - Model Performance', fontsize=16, fontweight='bold')

# ROC曲线
ax1 = axes[0, 0]
fpr, tpr, _ = roc_curve(test_labels, test_preds)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {ta:.4f}')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# PR曲线
ax2 = axes[0, 1]
prec_curve, rec_curve, _ = precision_recall_curve(test_labels, test_preds)
ax2.plot(rec_curve, prec_curve, 'g-', linewidth=2, label=f'F1 = {tf:.4f}')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# 训练损失
ax3 = axes[1, 0]
ax3.plot(history['train_loss'], 'r-', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 验证指标
ax4 = axes[1, 1]
ax4.plot(history['val_auc'], 'b-', linewidth=2, label='Validation AUC')
ax4.plot(history['val_f1'], 'g--', linewidth=2, label='Validation F1')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Validation Metrics', fontsize=14, fontweight='bold')
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/performance.png', dpi=150, bbox_inches='tight')
print("  Saved: models/performance.png")

# 性能图2: 预测分布和指标汇总
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Sepsis 12-Hour Early Warning System - Prediction Analysis', fontsize=16, fontweight='bold')

# 预测分布
ax5 = axes2[0]
neg_preds = test_preds[test_labels == 0]
pos_preds = test_preds[test_labels == 1]
ax5.hist(neg_preds, bins=50, alpha=0.6, color='green', label='Normal', density=True)
ax5.hist(pos_preds, bins=50, alpha=0.6, color='red', label='Sepsis Warning', density=True)
ax5.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
ax5.set_xlabel('Prediction Probability', fontsize=12)
ax5.set_ylabel('Density', fontsize=12)
ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# 指标柱状图
ax6 = axes2[1]
metrics = ['AUC', 'F1', 'Precision', 'Recall']
values = [ta, tf, tp, tr]
colors = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24']
bars = ax6.bar(metrics, values, color=colors, width=0.6)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax6.set_ylim(0, 1.1)
for bar, val in zip(bars, values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/prediction_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: models/prediction_analysis.png")

# 特征重要性图
fig3, ax7 = plt.subplots(figsize=(12, 6))
feature_importance = []
for f in FEATURES:
    pos_mean = pos_df[f].mean()
    neg_mean = neg_df[f].mean()
    diff = abs((pos_mean - neg_mean) / (neg_mean + 1e-8))
    feature_importance.append(diff)

# 归一化
max_imp = max(feature_importance)
feature_importance = [x / max_imp for x in feature_importance]

# 排序
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = [FEATURES[i] for i in sorted_idx]
sorted_importance = [feature_importance[i] for i in sorted_idx]

bars = ax7.barh(range(len(sorted_features)), sorted_importance, color='#60a5fa')
ax7.set_yticks(range(len(sorted_features)))
ax7.set_yticklabels([FEATURE_NAMES[f] for f in sorted_features])
ax7.set_xlabel('Feature Importance (Normalized)', fontsize=12)
ax7.set_title('Feature Importance - Clinical Indicators', fontsize=14, fontweight='bold')
ax7.invert_yaxis()

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax7.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
print("  Saved: models/feature_importance.png")

print("\n" + "="*70)
print("训练完成！")
print("="*70)
print(f"\n文件已保存:")
print(f"  - models/sepsis_model.pt (模型)")
print(f"  - models/norm_params.json (标准化参数)")
print(f"  - models/data_stats.json (数据统计)")
print(f"  - models/test_results.json (测试结果)")
print(f"  - models/performance.png (性能图)")
print(f"  - models/prediction_analysis.png (预测分析图)")
print(f"  - models/feature_importance.png (特征重要性图)")
