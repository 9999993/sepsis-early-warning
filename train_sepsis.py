"""
脓毒症12小时早期预警系统 - 数据生成与模型训练
基于MIMIC-III真实统计，生成合理数据并训练LSTM模型
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

# ==================== 配置 ====================
TOTAL_PATIENTS = 6000
SEPSIS_RATE = 0.15
MIN_HOURS = 24
MAX_HOURS = 60
SEQUENCE_LENGTH = 12  # 12小时预警窗口
FEATURES = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','BaseExcess','HCO3','FiO2','pH','PaCO2']
FEATURE_NAMES = {
    'HR':'心率','O2Sat':'血氧','Temp':'体温','SBP':'收缩压',
    'MAP':'平均压','DBP':'舒张压','Resp':'呼吸','BaseExcess':'碱剩余',
    'HCO3':'碳酸氢根','FiO2':'吸氧浓度','pH':'pH值','PaCO2':'二氧化碳'
}

# 基于MIMIC-III的ICU患者正常值
BASELINE = {
    'HR':82, 'O2Sat':97, 'Temp':36.8, 'SBP':120, 'MAP':80, 'DBP':65,
    'Resp':16, 'BaseExcess':-1, 'HCO3':24, 'FiO2':0.30, 'pH':7.39, 'PaCO2':39
}
VARIABILITY = {
    'HR':15, 'O2Sat':2, 'Temp':0.5, 'SBP':18, 'MAP':12, 'DBP':10,
    'Resp':4, 'BaseExcess':3, 'HCO3':3, 'FiO2':0.10, 'pH':0.04, 'PaCO2':6
}

print("="*70)
print("脓毒症12小时早期预警系统 - 训练")
print("="*70)

# ==================== 数据生成 ====================
print("\n[1/5] 生成6000名患者数据（15%脓毒症率）...")

def generate_patient(pid, will_sepsis):
    """生成单个患者的时序数据"""
    n_hours = np.random.randint(MIN_HOURS, MAX_HOURS + 1)
    patient_offset = {f: np.random.normal(0, VARIABILITY[f]*0.3) for f in FEATURES}
    
    # 脓毒症发作时间（第8-20小时）
    onset = np.random.randint(8, 20) if will_sepsis else None
    recovery = onset + np.random.randint(15, 25) if will_sepsis else None
    
    records = []
    for h in range(n_hours):
        record = {'patient_id': pid, 'hour': h}
        
        for f in FEATURES:
            base = BASELINE[f] + patient_offset[f]
            noise = np.random.normal(0, VARIABILITY[f] * 0.15)
            val = base + noise
            
            # 脓毒症恶化
            if will_sepsis and onset is not None and h > onset:
                hours_sick = h - onset
                prog = min(1.0, hours_sick / 12)
                fluc = np.random.normal(1.0, 0.15)
                
                if f == 'HR':
                    val = base * (1 + 0.30 * prog * fluc) + noise
                elif f == 'O2Sat':
                    val = base * (1 - 0.06 * prog * fluc) + noise
                elif f == 'Temp':
                    val = base + 1.0 * prog * fluc + noise * 0.5
                elif f == 'SBP':
                    val = base * (1 - 0.25 * prog * fluc) + noise
                elif f == 'MAP':
                    val = base * (1 - 0.28 * prog * fluc) + noise
                elif f == 'Resp':
                    val = base * (1 + 0.50 * prog * fluc) + noise
                elif f == 'pH':
                    val = base - 0.06 * prog * fluc + noise * 0.3
                elif f == 'BaseExcess':
                    val = base - 5 * prog * fluc + noise
            
            # 限制范围
            limits = {
                'HR':(40,160),'O2Sat':(75,100),'Temp':(35,41),'SBP':(60,180),
                'MAP':(40,120),'DBP':(30,100),'Resp':(8,40),'BaseExcess':(-15,15),
                'HCO3':(10,40),'FiO2':(0.21,1.0),'pH':(7.0,7.6),'PaCO2':(20,80)
            }
            val = max(limits[f][0], min(limits[f][1], val))
            record[f] = round(val, 2) if f in ['Temp','pH','FiO2'] else round(val, 1)
        
        # 12小时预警标签：未来12小时内会脓毒症
        if will_sepsis and onset is not None:
            # 从onset前12小时到recovery都标记为1
            if h >= onset - 12 and h <= recovery:
                record['label'] = 1
            else:
                record['label'] = 0
        else:
            record['label'] = 0
        
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

# ==================== 数据分割 ====================
print("\n[2/5] 按7:1.5:1.5分割数据集...")
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

# ==================== 标准化 ====================
print("\n[3/5] 计算标准化参数...")
norm_params = {'mean': train_df[FEATURES].mean().to_dict(), 'std': train_df[FEATURES].std().to_dict()}
os.makedirs('models', exist_ok=True)
with open('models/norm_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# 保存数据统计
data_stats = {
    'total_patients': TOTAL_PATIENTS,
    'sepsis_patients': sepsis_count,
    'sepsis_rate': SEPSIS_RATE,
    'train_patients': len(train_ids),
    'val_patients': len(val_ids),
    'test_patients': len(test_ids),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'train_positive': float(train_df['label'].mean()),
    'val_positive': float(val_df['label'].mean()),
    'test_positive': float(test_df['label'].mean()),
}
with open('models/data_stats.json', 'w') as f:
    json.dump(data_stats, f, indent=2)

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

train_ds = SepsisDataset(train_df, norm_params, SEQUENCE_LENGTH)
val_ds = SepsisDataset(val_df, norm_params, SEQUENCE_LENGTH)
test_ds = SepsisDataset(test_df, norm_params, SEQUENCE_LENGTH)
print(f"\n序列数据集:")
print(f"  训练: {len(train_ds)}")
print(f"  验证: {len(val_ds)}")
print(f"  测试: {len(test_ds)}")

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
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = Attention(hidden_dim)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
print("\n[5/5] 开始训练（40轮）...")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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

for epoch in range(40):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.squeeze().to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        print(f"  Epoch {epoch+1}/40: Loss={total_loss/len(train_loader):.4f}, AUC={va:.4f}, F1={vf:.4f}")

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
    'test_recall': float(tr),
}
with open('models/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ==================== 生成性能图 ====================
print("\n生成性能图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('脓毒症12小时预警系统 - 模型性能评估', fontsize=16, fontweight='bold')

# 1. ROC曲线
ax1 = axes[0, 0]
fpr, tpr, _ = roc_curve(test_labels, test_preds)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {ta:.4f}')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. 精确率-召回率曲线
ax2 = axes[0, 1]
precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_preds)
ax2.plot(recall_curve, precision_curve, 'g-', linewidth=2, label=f'F1 = {tf:.4f}')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. 训练损失曲线
ax3 = axes[1, 0]
ax3.plot(history['train_loss'], 'r-', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 验证AUC曲线
ax4 = axes[1, 1]
ax4.plot(history['val_auc'], 'b-', linewidth=2, label='Val AUC')
ax4.plot(history['val_f1'], 'g--', linewidth=2, label='Val F1')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Validation Metrics', fontsize=14, fontweight='bold')
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/performance.png', dpi=150, bbox_inches='tight')
print("  性能图已保存: models/performance.png")

# 生成第二个图：混淆矩阵和预测分布
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('脓毒症12小时预警系统 - 预测分析', fontsize=16, fontweight='bold')

# 5. 预测概率分布
ax5 = axes2[0]
neg_preds = test_preds[test_labels == 0]
pos_preds = test_preds[test_labels == 1]
ax5.hist(neg_preds, bins=50, alpha=0.6, color='green', label='正常', density=True)
ax5.hist(pos_preds, bins=50, alpha=0.6, color='red', label='脓毒症预警', density=True)
ax5.axvline(x=0.5, color='black', linestyle='--', label='阈值=0.5')
ax5.set_xlabel('预测概率', fontsize=12)
ax5.set_ylabel('密度', fontsize=12)
ax5.set_title('预测概率分布', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# 6. 性能指标柱状图
ax6 = axes2[1]
metrics = ['AUC', 'F1', 'Precision', 'Recall']
values = [ta, tf, tp, tr]
colors = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24']
bars = ax6.bar(metrics, values, color=colors, width=0.6)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('性能指标汇总', fontsize=14, fontweight='bold')
ax6.set_ylim(0, 1.1)
for bar, val in zip(bars, values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/prediction_analysis.png', dpi=150, bbox_inches='tight')
print("  预测分析图已保存: models/prediction_analysis.png")

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
