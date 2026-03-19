"""
重新训练 - 更平衡的12小时预警模型
目标：风险逐渐变化，不是突然跳变
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

np.random.seed(123)
torch.manual_seed(123)

TOTAL_PATIENTS = 5000
SEPSIS_RATE = 0.15
MIN_HOURS = 24
MAX_HOURS = 50
SEQUENCE_LENGTH = 12
FEATURES = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','BaseExcess','HCO3','FiO2','pH','PaCO2']

BASELINE = {'HR':82,'O2Sat':97,'Temp':36.8,'SBP':120,'MAP':80,'DBP':65,'Resp':16,'BaseExcess':-1,'HCO3':24,'FiO2':0.30,'pH':7.39,'PaCO2':39}
VAR = {'HR':15,'O2Sat':2,'Temp':0.5,'SBP':18,'MAP':12,'DBP':10,'Resp':4,'BaseExcess':3,'HCO3':3,'FiO2':0.10,'pH':0.04,'PaCO2':6}

print("="*70)
print("重新训练 - 平衡的12小时预警模型")
print("="*70)

def generate_patient(pid, will_sepsis):
    n_hours = np.random.randint(MIN_HOURS, MAX_HOURS + 1)
    offset = {f: np.random.normal(0, VAR[f]*0.3) for f in FEATURES}
    onset = np.random.randint(8, 18) if will_sepsis else None
    recovery = onset + np.random.randint(12, 20) if will_sepsis else None
    
    records = []
    for h in range(n_hours):
        record = {'patient_id': pid, 'hour': h}
        
        for f in FEATURES:
            base = BASELINE[f] + offset[f]
            noise = np.random.normal(0, VAR[f] * 0.2)  # 更大的正常波动
            val = base + noise
            
            # 脓毒症恶化 - 更渐进
            if will_sepsis and onset is not None and h >= onset:
                hours_sick = h - onset
                # 使用sigmoid让变化更平滑
                progress = min(1.0, hours_sick / 20)  # 20小时完全恶化
                sigmoid = 1 / (1 + np.exp(-8 * (progress - 0.5)))
                
                if f == 'HR':
                    val = base + 20 * sigmoid + np.random.normal(0, 8)
                elif f == 'O2Sat':
                    val = base - 4 * sigmoid + np.random.normal(0, 1.5)
                elif f == 'Temp':
                    val = base + 0.6 * sigmoid + np.random.normal(0, 0.3)
                elif f == 'SBP':
                    val = base - 20 * sigmoid + np.random.normal(0, 10)
                elif f == 'MAP':
                    val = base - 15 * sigmoid + np.random.normal(0, 8)
                elif f == 'Resp':
                    val = base + 8 * sigmoid + np.random.normal(0, 3)
                elif f == 'pH':
                    val = base - 0.04 * sigmoid + np.random.normal(0, 0.02)
                elif f == 'BaseExcess':
                    val = base - 3 * sigmoid + np.random.normal(0, 1.5)
            
            limits = {'HR':(40,160),'O2Sat':(80,100),'Temp':(35,41),'SBP':(60,180),'MAP':(40,120),'DBP':(30,100),'Resp':(8,40),'BaseExcess':(-15,15),'HCO3':(10,40),'FiO2':(0.21,1.0),'pH':(7.0,7.6),'PaCO2':(20,80)}
            val = max(limits[f][0], min(limits[f][1], val))
            record[f] = round(val, 2) if f in ['Temp','pH','FiO2'] else round(val, 1)
        
        # 12小时预警标签
        if will_sepsis and onset is not None:
            record['label'] = 1 if (h >= onset - 12 and h <= recovery) else 0
        else:
            record['label'] = 0
        
        records.append(record)
    return records

print("\n[1/5] 生成数据...")
all_records = []
sepsis_count = 0
for i in range(TOTAL_PATIENTS):
    will_sepsis = np.random.random() < SEPSIS_RATE
    if will_sepsis:
        sepsis_count += 1
    all_records.extend(generate_patient(f'P{i:05d}', will_sepsis))
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{TOTAL_PATIENTS}")

df = pd.DataFrame(all_records)
print(f"\n脓毒症: {sepsis_count}/{TOTAL_PATIENTS} ({sepsis_count/TOTAL_PATIENTS*100:.1f}%), 阳性率: {df['label'].mean()*100:.1f}%")

print("\n[2/5] 分割...")
pids = df['patient_id'].unique()
np.random.shuffle(pids)
n = len(pids)
train_ids = pids[:int(n*0.7)]
val_ids = pids[int(n*0.7):int(n*0.85)]
test_ids = pids[int(n*0.85):]

train_df = df[df['patient_id'].isin(train_ids)]
val_df = df[df['patient_id'].isin(val_ids)]
test_df = df[df['patient_id'].isin(test_ids)]

print(f"训练: {len(train_ids)}, 验证: {len(val_ids)}, 测试: {len(test_ids)}")

print("\n[3/5] 标准化...")
norm_params = {'mean': train_df[FEATURES].mean().to_dict(), 'std': train_df[FEATURES].std().to_dict()}
os.makedirs('models', exist_ok=True)
with open('models/norm_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

class SepsisDataset(Dataset):
    def __init__(self, df, norm, seq_len=12):
        self.seq_len = seq_len
        self.data = df.copy()
        for feat in FEATURES:
            self.data[feat] = (self.data[feat]-norm['mean'][feat])/(norm['std'][feat]+1e-8)
        self.seqs, self.labels = [], []
        for pid in self.data['patient_id'].unique():
            p = self.data[self.data['patient_id']==pid].sort_values('hour')
            if len(p) >= seq_len:
                vals = p[FEATURES].values
                labs = p['label'].values
                for i in range(len(p)-seq_len+1):
                    self.seqs.append(vals[i:i+seq_len])
                    self.labels.append(labs[i+seq_len-1])
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return torch.FloatTensor(self.seqs[i]), torch.FloatTensor([self.labels[i]])

train_ds = SepsisDataset(train_df, norm_params)
val_ds = SepsisDataset(val_df, norm_params)
test_ds = SepsisDataset(test_df, norm_params)
print(f"序列: 训练{len(train_ds)}, 验证{len(val_ds)}, 测试{len(test_ds)}")

print("\n[4/5] 模型...")

class Attention(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(hs*2,hs),nn.Tanh(),nn.Linear(hs,1))
    def forward(self, x):
        w = torch.softmax(self.a(x),dim=1)
        return torch.sum(w*x,dim=1), w.squeeze(-1)

class SepsisLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(12)
        self.lstm = nn.LSTM(12, 64, 2, batch_first=True, bidirectional=True, dropout=0.4)
        self.attn = Attention(64)
        self.clf = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),  # 更强的dropout
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x, ret=False):
        x = self.norm(x)
        o,_ = self.lstm(x)
        c,w = self.attn(o)
        out = self.clf(c).squeeze(-1)
        return (out,w) if ret else out

model = SepsisLSTM()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

print("\n[5/5] 训练...")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)  # 更强的正则化
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
criterion = nn.BCELoss()

def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(DEVICE), y.to(DEVICE)
            p = model(X)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, (preds>0.5).astype(int), zero_division=0)
    prec = precision_score(labels, (preds>0.5).astype(int), zero_division=0)
    rec = recall_score(labels, (preds>0.5).astype(int), zero_division=0)
    return auc, f1, prec, rec, preds, labels

history = {'train_loss':[],'val_auc':[],'val_f1':[]}
best_auc = 0

for epoch in range(30):
    model.train()
    total_loss = 0
    for X,y in train_loader:
        X,y = X.to(DEVICE), y.squeeze().to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    va, vf, vp, vr, _, _ = evaluate(val_loader)
    
    history['train_loss'].append(total_loss/len(train_loader))
    history['val_auc'].append(va)
    history['val_f1'].append(vf)
    
    if va > best_auc:
        best_auc = va
        torch.save({'model_state_dict':model.state_dict(),'val_auc':va,'epoch':epoch}, 'models/sepsis_model.pt')
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, AUC={va:.4f}, F1={vf:.4f}")

print("\n" + "="*70)
print("测试集评估")
print("="*70)
ckpt = torch.load('models/sepsis_model.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
ta, tf, tp, tr, test_preds, test_labels = evaluate(test_loader)

print(f"验证AUC: {best_auc:.4f}")
print(f"测试集: AUC={ta:.4f}, F1={tf:.4f}, Precision={tp:.4f}, Recall={tr:.4f}")

# 保存统计
data_stats = {
    'total_patients': TOTAL_PATIENTS,
    'sepsis_patients': sepsis_count,
    'train_patients': len(train_ids),
    'val_patients': len(val_ids),
    'test_patients': len(test_ids),
}
with open('models/data_stats.json', 'w') as f:
    json.dump(data_stats, f, indent=2)

results = {'val_auc':float(best_auc),'test_auc':float(ta),'test_f1':float(tf),'test_precision':float(tp),'test_recall':float(tr)}
with open('models/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 生成性能图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('脓毒症12小时预警系统 - 模型性能', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
fpr, tpr, _ = roc_curve(test_labels, test_preds)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {ta:.4f}')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
prec_curve, rec_curve, _ = precision_recall_curve(test_labels, test_preds)
ax2.plot(rec_curve, prec_curve, 'g-', linewidth=2, label=f'F1 = {tf:.4f}')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(history['train_loss'], 'r-', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.plot(history['val_auc'], 'b-', linewidth=2, label='Val AUC')
ax4.plot(history['val_f1'], 'g--', linewidth=2, label='Val F1')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Score')
ax4.set_title('Validation Metrics')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/performance.png', dpi=150, bbox_inches='tight')
print("性能图已保存")

# 预测分布图
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
ax5 = axes2[0]
neg = test_preds[test_labels == 0]
pos = test_preds[test_labels == 1]
ax5.hist(neg, bins=50, alpha=0.6, color='green', label='正常', density=True)
ax5.hist(pos, bins=50, alpha=0.6, color='red', label='脓毒症预警', density=True)
ax5.axvline(x=0.5, color='black', linestyle='--', label='阈值=0.5')
ax5.set_xlabel('预测概率')
ax5.set_ylabel('密度')
ax5.set_title('预测概率分布')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = axes2[1]
metrics = ['AUC', 'F1', 'Precision', 'Recall']
values = [ta, tf, tp, tr]
colors = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24']
bars = ax6.bar(metrics, values, color=colors, width=0.6)
ax6.set_ylabel('Score')
ax6.set_title('性能指标')
ax6.set_ylim(0, 1.1)
for bar, val in zip(bars, values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center')

plt.tight_layout()
plt.savefig('models/prediction_analysis.png', dpi=150, bbox_inches='tight')
print("预测分析图已保存")

print("\n完成!")
