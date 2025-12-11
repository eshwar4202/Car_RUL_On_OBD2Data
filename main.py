import os, json, stat, shutil, math, pickle
import numpy as np, pandas as pd
from glob import glob
from scipy.interpolate import interp1d
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Config
DATA_DIR = "/content/obd2data"
KAGGLE_USERNAME = "eshwar4202"
KAGGLE_KEY = "KGAT_eb816d5a51f66f01aacbb1cedbb6972d" 

DATASET_SLUG = "eron93br/obd2data"
PREFERRED_SENSORS = [
    "ENGINE_RPM ()", "ENGINE_RUN_TINE ()", "VEHICLE_SPEED ()", "THROTTLE ()",
    "ENGINE_LOAD ()", "COOLANT_TEMPERATURE ()", "INTAKE_MANIFOLD_PRESSURE ()",
    "MAF ()", "INTAKE_AIR_TEMP ()", "TIMING_ADVANCE ()"
]

PCT_MINOR = 0.95
PCT_SEVERE = 0.99
CLASS_WEIGHTS = {"healthy": 0.0, "minor": 1.0, "severe": 3.0, "critical": 5.0}

WINDOW_SIZE = 50
WINDOW_STRIDE = 5
RUL_MAX_CYCLES = 1000
RANDOM_SEED = 42
TRAIN_EPOCHS = 40
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if not os.path.exists(DATA_DIR) or len(glob(os.path.join(DATA_DIR,"*.csv"))) == 0:
    print("No dataset found locally. Attempting Kaggle download...")
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise RuntimeError("Dataset not found in /content/obd2data. To enable automatic download set KAGGLE_USERNAME and KAGGLE_KEY in the script.")
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json","w") as f:
        json.dump({"username":KAGGLE_USERNAME,"key":KAGGLE_KEY}, f)
    os.chmod("/root/.kaggle/kaggle.json", stat.S_IRUSR | stat.S_IWUSR)
  
    !kaggle datasets download -d {DATASET_SLUG} -p /content --unzip
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in glob("/content/*.csv"):
        shutil.move(f, os.path.join(DATA_DIR, os.path.basename(f)))
    for f in glob("/content/*.zip"):
        try:
            shutil.unpack_archive(f, DATA_DIR)
        except Exception:
            pass
    os.remove("/root/.kaggle/kaggle.json")
    print("Downloaded and moved dataset to", DATA_DIR)


csv_files = sorted(glob(os.path.join(DATA_DIR, "live*.csv")))
if len(csv_files) == 0:
    csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

print(f"Found {len(csv_files)} CSV files. First few: {csv_files[:5]}")
frames = []
for i, fpath in enumerate(csv_files):
    df_part = pd.read_csv(fpath)
    df_part.columns = [c.strip() for c in df_part.columns]
    df_part["_unit_id"] = os.path.splitext(os.path.basename(fpath))[0]
    if "Time" not in df_part.columns and "TIME" not in df_part.columns and "Time " not in df_part.columns:
        df_part["Time"] = np.arange(len(df_part))
    frames.append(df_part)
df = pd.concat(frames, ignore_index=True)
print("Concatenated dataframe shape:", df.shape)
UNIT_COL = "_unit_id"
TIME_COL = "Time"

sensors = [c for c in PREFERRED_SENSORS if c in df.columns]
if len(sensors) < 3:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    sensors = [c for c in numeric if c not in [UNIT_COL, TIME_COL]]
print("Using sensor columns:", sensors[:20])
if len(sensors) < 2:
    raise RuntimeError("Not enough numeric sensor columns found. Edit PREFERRED_SENSORS or upload proper CSVs.")


nan_frac = df[sensors].isna().mean()
drop_cols = nan_frac[nan_frac > 0.3].index.tolist()
if drop_cols:
    print("Dropping columns with >30% NaN:", drop_cols)
    sensors = [s for s in sensors if s not in drop_cols]

print("Applying per-unit linear interpolation + ffill/bfill for sensor cols...")
for u in df[UNIT_COL].unique():
    mask = df[UNIT_COL] == u
    sub = df.loc[mask, sensors]
    sub_interp = sub.interpolate(method='linear', limit_direction='both', axis=0)
    sub_interp = sub_interp.ffill().bfill()
    for c in sensors:
        if sub_interp[c].isna().all():
            sub_interp[c] = df[c].median()
    df.loc[mask, sensors] = sub_interp.values
print("Interpolation done. Remaining NaNs (total):", int(df[sensors].isna().sum().sum()))

print("Building auto threshold rules (percentile-based)...")
auto_rules = []
for col in sensors:
    vals = df[col].dropna()
    if len(vals) < 50:
        continue
    thr_minor = float(vals.quantile(PCT_MINOR))
    thr_severe = float(vals.quantile(PCT_SEVERE))
    if thr_severe > thr_minor:
        auto_rules.append((col, "gt", thr_severe, "severe"))
        auto_rules.append((col, "gt", thr_minor, "minor"))

print("Auto rules sample (first 8):", auto_rules[:8])

def apply_rules_row(row, rules):
    matched = []
    for col, op, thr, lab in rules:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v): 
            continue
        if op == "gt" and v > thr:
            matched.append(lab)
        if op == "lt" and v < thr:
            matched.append(lab)
    if not matched:
        return "healthy"
    priority = {"healthy":0,"minor":1,"severe":2,"critical":3}
    return sorted(matched, key=lambda x: priority.get(x,1), reverse=True)[0]

print("Applying auto-rules to create fault_label...")
df["fault_label"] = df.apply(lambda r: apply_rules_row(r, auto_rules), axis=1)
print("Label distribution:", df["fault_label"].value_counts().to_dict())

non_healthy_frac = (df["fault_label"] != "healthy").mean()
if non_healthy_frac < 0.03:
    print("Few non-healthy labels (", non_healthy_frac, ") -> relaxing thresholds to 90/97 percentiles")
    PCT_MINOR2, PCT_SEVERE2 = 0.90, 0.97
    auto_rules = []
    for col in sensors:
        vals = df[col].dropna()
        if len(vals) < 50:
            continue
        thr_minor = float(vals.quantile(PCT_MINOR2))
        thr_severe = float(vals.quantile(PCT_SEVERE2))
        if thr_severe > thr_minor:
            auto_rules.append((col, "gt", thr_severe, "severe"))
            auto_rules.append((col, "gt", thr_minor, "minor"))
    df["fault_label"] = df.apply(lambda r: apply_rules_row(r, auto_rules), axis=1)
    print("New label distribution:", df["fault_label"].value_counts().to_dict())
def create_windows_for_unit(unit_df, sensors, win, stride):
    arr = unit_df[sensors].values
    n = arr.shape[0]
    starts = list(range(0, max(1, n - win + 1), stride))
    Xw, ends = [], []
    if len(starts) == 0:
        pad = np.repeat(arr[-1:].reshape(1, -1), win, axis=0)
        return np.expand_dims(pad, 0), [n-1]
    for s in starts:
        Xw.append(arr[s:s+win])
        ends.append(s + win - 1)
    return np.stack(Xw), ends

X_windows_list, y_windows_list, window_meta = [], [], []
units = df[UNIT_COL].unique().tolist()
for u in units:
    unit_df = df[df[UNIT_COL]==u].sort_values(TIME_COL).reset_index(drop=True)
    Xw, ends = create_windows_for_unit(unit_df, sensors, WINDOW_SIZE, WINDOW_STRIDE)
    labels_arr = unit_df["fault_label"].values
    y_w = []
    for s in range(0, max(1, unit_df.shape[0]-WINDOW_SIZE+1), WINDOW_STRIDE):
        mode = pd.Series(labels_arr[s:s+WINDOW_SIZE]).mode()
        y_w.append(mode.iloc[0] if not mode.empty else labels_arr[-1])
    if len(y_w) == 0:
        y_w = [labels_arr[-1]]
    X_windows_list.append(Xw)
    y_windows_list.append(np.array(y_w))
    window_meta.extend([(u, end) for end in ends])

X_windows = np.concatenate(X_windows_list, axis=0)
y_windows = np.concatenate(y_windows_list, axis=0)
print("Window-level shape:", X_windows.shape, "label counts:", Counter(y_windows))

def summarize_windows(Xw):
    feats = []
    for w in Xw:
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        mn = w.min(axis=0)
        mx = w.max(axis=0)
        feats.append(np.concatenate([mean, std, mn, mx]))
    return np.array(feats)

X_feats = summarize_windows(X_windows)
le = LabelEncoder()
y_enc = le.fit_transform(y_windows)
class_names = le.classes_.tolist()
print("Classifier classes:", class_names)

sc_clf = StandardScaler()
X_feats_s = sc_clf.fit_transform(X_feats)

X_tr, X_val, y_tr, y_val = train_test_split(X_feats_s, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc)
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced")
print("Training RandomForest classifier...")
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_val)
print("Classifier report:\n", classification_report(y_val, y_pred, target_names=class_names))
probs = clf.predict_proba(sc_clf.transform(X_feats)) 

sorted_by_weight = sorted(CLASS_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True)
worst_class = sorted_by_weight[0][0]
classes_list = le.inverse_transform(np.arange(len(class_names))).tolist()
if worst_class in classes_list:
    worst_idx = classes_list.index(worst_class)
    severity_windows = probs[:, worst_idx]
    print("Severity = prob(worst_class). Stats:", severity_windows.min(), severity_windows.max(), severity_windows.mean())
else:
    weights_vec = np.array([CLASS_WEIGHTS.get(c,1.0) for c in classes_list]) * 2.5
    severity_windows = probs.dot(weights_vec)
    print("Severity (weighted) stats:", severity_windows.min(), severity_windows.max(), severity_windows.mean())

windows_df = pd.DataFrame(window_meta, columns=[UNIT_COL, "end_index"])
windows_df["severity"] = severity_windows

df["severity"] = np.nan
for u in units:
    unit_idx = df[df[UNIT_COL]==u].index.to_numpy()
    unit_df = df[df[UNIT_COL]==u].sort_values(TIME_COL).reset_index()
    wmask = windows_df[windows_df[UNIT_COL]==u]
    if wmask.shape[0] == 0:
        df.loc[unit_df['index'], "severity"] = 0.0
        continue
    x_pts = wmask["end_index"].values
    y_pts = wmask["severity"].values
    if len(x_pts) == 1:
        df.loc[unit_df['index'], "severity"] = y_pts[0]
        continue
    f = interp1d(x_pts, y_pts, bounds_error=False, fill_value=(y_pts[0], y_pts[-1]))
    mapped = f(unit_df.index.values)
    df.loc[unit_df['index'], "severity"] = mapped

df["severity"].ffill(inplace=True); df["severity"].bfill(inplace=True); df["severity"].fillna(0.0, inplace=True)
print("Row-level severity stats:", df["severity"].min(), df["severity"].max(), df["severity"].mean())

# ----------------- Generate synthetic RUL per-unit (normalize per unit and invert) -----------------
df["synthetic_RUL"] = np.nan
for u in units:
    idxs = df[df[UNIT_COL]==u].index
    sev = df.loc[idxs, "severity"].values
    if len(sev)==0:
        continue
    mn, mx = float(sev.min()), float(sev.max())
    if mx - mn < 1e-9:
        norm = np.zeros_like(sev)
    else:
        norm = (sev - mn) / (mx - mn)
    df.loc[idxs, "synthetic_RUL"] = (1.0 - norm) * RUL_MAX_CYCLES

print("Synthetic RUL overall stats:", df["synthetic_RUL"].describe().to_dict())

# ----------------- Build RUL regression windows & targets -----------------
X_rul_list, y_rul_list = [], []
for u in units:
    unit_df = df[df[UNIT_COL]==u].sort_values(TIME_COL).reset_index(drop=True)
    Xw, ends = create_windows_for_unit(unit_df, sensors, WINDOW_SIZE, WINDOW_STRIDE)
    if len(ends) == 0:
        continue
    targets = unit_df.loc[ends, "synthetic_RUL"].values
    # drop nan targets
    mask_valid = ~np.isnan(targets)
    if mask_valid.sum() == 0:
        continue
    Xw = Xw[mask_valid]; targets = targets[mask_valid]
    X_rul_list.append(Xw); y_rul_list.append(targets)

if len(X_rul_list) == 0:
    raise RuntimeError("No RUL windows could be built. Inspect dataset or adjust WINDOW_SIZE/STRIDE.")

X_rul = np.concatenate(X_rul_list, axis=0)
y_rul = np.concatenate(y_rul_list, axis=0)
print("RUL dataset shapes:", X_rul.shape, y_rul.shape)

# ----------------- Clean windows (drop windows with any NaN, impute if small fraction left) -----------------
# Remove windows with any NaNs
nan_mask = np.isnan(X_rul).any(axis=(1,2))
if nan_mask.any():
    print("Dropping", nan_mask.sum(), "windows with NaNs in sensors")
    keep = ~nan_mask
    X_rul = X_rul[keep]; y_rul = y_rul[keep]
# final check
if X_rul.shape[0] == 0:
    raise RuntimeError("All windows dropped after NaN removal. Consider smaller window or different sensors.")

# ----------------- Scale targets (MinMax) and split -----------------
y_rul = y_rul.reshape(-1,1)
y_scaler = MinMaxScaler(feature_range=(0,1))
y_rul_scaled = y_scaler.fit_transform(y_rul).reshape(-1)

X_train, X_temp, y_train, y_temp = train_test_split(X_rul, y_rul_scaled, test_size=0.30, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

# ----------------- scale input features (fit on train) -----------------
ns, wlen, nfeat = X_train.shape
scaler_rul = StandardScaler()
scaler_rul.fit(X_train.reshape(-1, nfeat))

def scale_windows(X):
    flat = X.reshape(-1, X.shape[2])
    flat_s = scaler_rul.transform(flat)
    return flat_s.reshape(X.shape)

X_train_s = scale_windows(X_train)
X_val_s = scale_windows(X_val)
X_test_s = scale_windows(X_test)

print("Train/Val/Test shapes:", X_train_s.shape, X_val_s.shape, X_test_s.shape)

# ----------------- Build CNN-LSTM model -----------------
def build_cnn_lstm(window_size, n_features):
    inp = layers.Input(shape=(window_size, n_features))
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    out = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    return model

model = build_cnn_lstm(WINDOW_SIZE, X_train_s.shape[2])
model.summary()

# ----------------- Train RUL model -----------------
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
ckpt = ModelCheckpoint("best_rul_model_final.h5", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
                    epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, ckpt], verbose=1)

# ----------------- Evaluate -----------------
model = tf.keras.models.load_model("best_rul_model_final.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
y_pred_s = model.predict(X_test_s).reshape(-1)
# invert scaling
y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).reshape(-1)
y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(f"Final Test RMSE: {rmse:.3f}, R2: {r2:.3f}")

# plots
nplot = min(300, len(y_true))
plt.figure(figsize=(12,4)); plt.plot(y_true[:nplot], label='true'); plt.plot(y_pred[:nplot], label='pred'); plt.legend(); plt.title("RUL True vs Pred"); plt.show()
plt.figure(figsize=(10,4)); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val'); plt.yscale('log'); plt.legend(); plt.title("Train/Val Loss"); plt.show()

# ----------------- Save artifacts -----------------
model.save("final_rul_model_eron93br.h5")
with open("scaler_rul_eron93br.pkl", "wb") as f:
    pickle.dump(scaler_rul, f)
with open("fault_clf_eron93br.pkl", "wb") as f:
    pickle.dump({"clf":clf, "scaler":sc_clf, "label_encoder":le, "sensors":sensors, "y_scaler":y_scaler}, f)

print("Saved: final_rul_model_eron93br.h5, best_rul_model_final.h5, fault_clf_eron93br.pkl, scaler_rul_eron93br.pkl")
