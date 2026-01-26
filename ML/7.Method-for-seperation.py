# app.py
import os, math, warnings, base64, io
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, chardet, lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

# Model directories
AM_I_II_MODELS_DIR = './svr-models'
AM_III_MODELS_DIR = './lgb-models'
AM_IV_VII_MODELS_DIR = './svr-model-other4'
SMARTS_FILE = './SMARTS/priority_fgs_823_newnew.txt'
FEATURE_COLS=['MolWt','logP','TPSA','H_bond_donors','H_bond_acceptors']
FP_COLS=[f'col{i}' for i in range(823)]
MG_COLS=[f'fp_{i}' for i in range(1024)]
ALL_FEATURES=FEATURE_COLS+FP_COLS+MG_COLS
# Fixed method order: AM-I, AM-II, AM-III, AM-IV, AM-V, AM-VI, AM-VII
FIXED_METHOD_ORDER=['AM-I','AM-II','AM-III','AM-IV','AM-V','AM-VI','AM-VII']

# Model file names mapping
MODEL_FILE_MAPPING = {
    'AM-I': {
        'model': './AM-I-svr-model/AM-I-filtered_with_labels_k4_svr_model.joblib',
        'scaler': './AM-I-svr-model/AM-I-filtered_with_labels_k4_scaler.joblib',
        'path': AM_I_II_MODELS_DIR
    },
    'AM-II': {
        'model': './AM-II-svr-model/AM-II-filtered_with_labels_k3_svr_model.joblib',
        'scaler': './AM-II-svr-model/AM-II-filtered_with_labels_k3_scaler.joblib',
        'path': AM_I_II_MODELS_DIR
    },
    'AM-III': {
        'model': 'AM-III-filtered_with_labels_k4_lgb.txt',
        'imputer': 'AM-III-filtered_with_labels_k4_imputer.pkl',
        'feature_list': 'AM-III-filtered_with_labels_k4_feature_list.pkl',
        'path': AM_III_MODELS_DIR
    },
    'AM-IV': {
        'model': 'AM-IV-filtered_final_svr_model.joblib',
        'scaler': 'AM-IV-filtered_final_scaler.joblib',
        'path': AM_IV_VII_MODELS_DIR
    },
    'AM-V': {
        'model': 'AM-V-filtered_final_svr_model.joblib',
        'scaler': 'AM-V-filtered_final_scaler.joblib',
        'path': AM_IV_VII_MODELS_DIR
    },
    'AM-VI': {
        'model': 'AM-VI-filtered_final_svr_model.joblib',
        'scaler': 'AM-VI-filtered_final_scaler.joblib',
        'path': AM_IV_VII_MODELS_DIR
    },
    'AM-VII': {
        'model': 'AM-VII-filtered_final_svr_model.joblib',
        'scaler': 'AM-VII-filtered_final_scaler.joblib',
        'path': AM_IV_VII_MODELS_DIR
    }
}

# Load SMARTS patterns
with open(SMARTS_FILE,'rb') as f:
    raw=f.read()
    enc=chardet.detect(raw)['encoding'] or 'utf-8'
with open(SMARTS_FILE,encoding=enc,errors='ignore') as f:
    SMARTS_PATTERNS=[l.strip() for l in f if l.strip()]

def calc_features(smiles:str)->Optional[np.ndarray]:
    """Calculate molecular features from SMILES string."""
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    base=[Descriptors.MolWt(mol),Descriptors.MolLogP(mol),Descriptors.TPSA(mol),
          Descriptors.NumHDonors(mol),Descriptors.NumHAcceptors(mol)]
    fp_823=[0]*823
    for i,sma in enumerate(SMARTS_PATTERNS):
        patt=Chem.MolFromSmarts(sma)
        if patt and mol.HasSubstructMatch(patt):
            fp_823[i]=1
    mg=AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=1024)
    return np.array(base+fp_823+list(mg),dtype=np.float32)

class ModelHub:
    """Manages loading and prediction using different method models."""
    def __init__(self):
        self.models:Dict[str,Any]={}
        self.scalers:Dict[str,Any]={}
        self._load()
    
    def _load(self):
        """Load models and scalers according to the mapping."""
        for method_name, file_info in MODEL_FILE_MAPPING.items():
            model_path = os.path.join(file_info['path'], file_info['model'])
            
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for {method_name}: {model_path}")
                continue
            
            # Handle AM-III (LightGBM) separately
            if method_name == 'AM-III':
                # Load LightGBM model
                self.models[method_name] = lgb.Booster(model_file=model_path)
                # Load imputer and feature list
                imp_path = os.path.join(file_info['path'], file_info['imputer'])
                feat_path = os.path.join(file_info['path'], file_info['feature_list'])
                
                if os.path.exists(imp_path) and os.path.exists(feat_path):
                    self.scalers[method_name] = (joblib.load(imp_path), joblib.load(feat_path))
                else:
                    print(f"Warning: Imputer or feature list not found for {method_name}")
                    self.scalers[method_name] = None
            else:
                # Load SVR models for other methods
                self.models[method_name] = joblib.load(model_path)
                # Load scaler
                scaler_path = os.path.join(file_info['path'], file_info['scaler'])
                if os.path.exists(scaler_path):
                    self.scalers[method_name] = joblib.load(scaler_path)
                else:
                    print(f"Warning: Scaler not found for {method_name}: {scaler_path}")
                    self.scalers[method_name] = None
    
    def predict(self,smiles:str)->Dict[str,Optional[float]]:
        """Predict retention time for a given SMILES using all methods."""
        feat=calc_features(smiles)
        if feat is None: return {m:None for m in self.models}
        
        base=feat[:5]
        rest=feat[5:]
        preds={}
        
        for name,model in self.models.items():
            if name not in self.scalers or self.scalers[name] is None:
                preds[name]=None
                continue
                
            if name=='AM-III':  # LightGBM model
                imp,feat_list=self.scalers[name]
                df=pd.DataFrame([feat],columns=ALL_FEATURES)
                Xsub=df[feat_list].values
                Xsub=imp.transform(Xsub)
                preds[name]=float(model.predict(Xsub)[0])
            else:  # SVR models
                scaler=self.scalers[name]
                base_scaled=scaler.transform(base.reshape(1,-1))[0]
                full=np.concatenate((base_scaled,rest)).reshape(1,-1)
                try:
                    preds[name]=float(model.predict(full)[0])
                except Exception as e:
                    print(f"Prediction error for {name}: {e}")
                    preds[name]=None
        return preds

class DataEvaluationSystem:
    """Evaluates datasets based on interval and range criteria."""
    def __init__(self,
                 min_interval=10,
                 value_range=(30, 120),
                 distance_weight=10,
                 range_weight=0.6,
                 importance_weight=5.0,
                 strict_penalty=True):
        self.min_interval = min_interval
        self.distance_weight = distance_weight
        self.range_weight = range_weight
        self.importance_weight = importance_weight
        self.strict_penalty = strict_penalty
        # Define ranges for specific methods
        self.per_method_range = {
            'AM-III': (30, 150),
            'AM-IV': (30, 150)
        }
        self.default_range = value_range  # Save default range

    # -------------------------------------------------
    # Internal utilities
    # -------------------------------------------------
    def _get_range(self, method: str) -> Tuple[int, int]:
        return self.per_method_range.get(method, self.default_range)

    def _calculate_interval_score(self, values: List[float]) -> Tuple[float, List[Dict]]:
        violations, penalty = [], 0
        sorted_vals = sorted(values)
        n = len(values)
        for i in range(n - 1):
            gap = sorted_vals[i + 1] - sorted_vals[i]
            if gap < self.min_interval:
                shortage = self.min_interval - gap
                w1 = (1 / (values.index(sorted_vals[i]) + 1)) ** 3
                w2 = (1 / (values.index(sorted_vals[i + 1]) + 1)) ** 3
                w = max(w1, w2) * self.importance_weight
                p = shortage * w * self.distance_weight / n
                penalty += p
                violations.append({
                    'type': 'interval',
                    'values': [sorted_vals[i], sorted_vals[i + 1]],
                    'required': self.min_interval,
                    'actual': gap,
                    'penalty': p
                })
        return penalty, violations

    def _calculate_range_score(self, values: List[float], method: str) -> Tuple[float, List[Dict]]:
        violations, penalty = [], 0
        min_v, max_v = self._get_range(method)
        for idx, val in enumerate(values):
            if val < min_v or val > max_v:
                importance = idx + 1
                imp_w = math.exp(-0.5 * (importance - 1)) * self.importance_weight
                dist = (min_v - val) if val < min_v else (val - max_v)
                p = dist * imp_w * self.range_weight / len(values)
                penalty += p
                violations.append({
                    'type': 'range',
                    'value': val,
                    'importance': importance,
                    'distance': dist,
                    'penalty': p
                })
                if idx == 0 and self.strict_penalty:
                    return -1, violations
        return penalty, violations

    def _normalize_score(self, d_pen: float, r_pen: float, n: int) -> float:
        if r_pen == -1:
            return -1
        max_d = self.min_interval * (n - 1) * sum(1 / i for i in range(1, n + 1))
        max_r = max(abs(self.default_range[0]), abs(self.default_range[1])) * sum(1 / i for i in range(1, n + 1))
        total_pen = self.distance_weight * d_pen + self.range_weight * r_pen
        max_total = self.distance_weight * max_d + self.range_weight * max_r
        return max(0, 1 - total_pen / max_total) if max_total else 1.0

    # -------------------------------------------------
    # Public interface
    # -------------------------------------------------
    def evaluate_single_dataset(self, values: List[float], method: str) -> Dict[str, Any]:
        d_pen, d_vio = self._calculate_interval_score(values)
        r_pen, r_vio = self._calculate_range_score(values, method)
        score = self._normalize_score(d_pen, r_pen, len(values))
        return {'values': values, 'final_score': score}

    def evaluate_datasets(self, datasets: List[List[float]], method_names: List[str]) -> List[Dict[str, Any]]:
        results = []
        for idx, data in enumerate(datasets):
            res = self.evaluate_single_dataset(data, method_names[idx])
            res['dataset_id'] = idx
            res['method_name'] = method_names[idx]
            results.append(res)
        return results

class PredictRequest(BaseModel):
    smiles_list:List[str]=Field(...,min_items=2,description="Input SMILES in order of importance (highest to lowest, ≥2 required)")

app=FastAPI(title="UPLC RT Scoring API")

hub=ModelHub()
eval_sys=DataEvaluationSystem(min_interval=10,value_range=(30,120),distance_weight=10,range_weight=0.6,importance_weight=5.0,strict_penalty=True)

def plot_to_b64(results:List[Dict])->str:
    """Generate visualization plot and convert to base64 string."""
    n=len(results)
    max_vals=max(len(r['values'])for r in results)
    fig,ax=plt.subplots(figsize=(14,max(5,n*1.1)))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    colors=plt.cm.tab10.colors
    
    # Determine coordinate ranges based on method
    xlims=[]
    for r in results:
        method=r['method_name']
        if method in {'AM-III','AM-IV'}:
            gmin, gmax = 20, 170
            span_min, span_max = 30, 150
        else:
            gmin, gmax = 20, 150
            span_min, span_max = 30, 120
        xlims.append((gmin,gmax,span_min,span_max))
    
    gmin_all=min(x[0] for x in xlims)
    gmax_all=max(x[1] for x in xlims)
    span_min_all=30
    span_max_all=max(x[3] for x in xlims)
    
    # Draw background color band (30-150)
    ax.axvspan(span_min_all,span_max_all,color='#E5E5EA',alpha=0.5,zorder=0)
    
    # Plot data points for each method
    for i,res in enumerate(results):
        method=res['method_name']
        if method in {'AM-III','AM-IV'}:
            gmin, gmax = 20, 170
            span_min, span_max = 30, 150
        else:
            gmin, gmax = 20, 150
            span_min, span_max = 30, 120
        
        y=n-i-1  # Top to bottom: AM-I to AM-VII
        vals=res['values']
        
        for j,v in enumerate(vals):
            size,color=200/(j+1),colors[j%10]
            marker='o' if span_min<=v<=span_max else'X'
            ax.scatter(v,y,s=size,c=[color],marker=marker,alpha=0.9,edgecolors='k',zorder=3)
        
        # Draw red lines for interval violations
        sorted_vals=sorted(vals)
        for k in range(len(sorted_vals)-1):
            if sorted_vals[k+1]-sorted_vals[k]<10:
                ax.plot(sorted_vals[k:k+2],[y,y],'r-',lw=3,alpha=0.7,zorder=2)
        
        # Add score text
        score_txt=f"{res['final_score']:.3f}" if res['final_score']>=0 else"Penalty"
        ax.text(gmax_all-0.02*(gmax_all-gmin_all),y,score_txt,ha='right',va='center',fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.9))
    
    ax.set_xlim(gmin_all,gmax_all)
    ax.set_ylim(-0.5,n-0.5)
    ax.set_xlabel('Retention Time (s)',fontsize=13)
    ax.set_ylabel('UPLC Method',fontsize=13)
    ax.set_title('Multi-Method Scoring Comparison',fontsize=14,fontweight='bold')
    ax.set_yticks(range(n))
    
    # Set method labels (AM-I at top, AM-VII at bottom)
    method_labels = [r['method_name'] for r in results]  # Results are in order AM-I to AM-VII
    ax.set_yticklabels(method_labels, fontsize=11)
    ax.set_xticklabels(ax.get_xticks(), fontsize=11)
    ax.grid(axis='x',linestyle='--',alpha=0.3)

    # Generate legend
    labels = ['P'] + [f'S{i+1}' for i in range(max_vals-1)]
    legend=[plt.scatter([],[],s=200/(j+1),c=[colors[j%10]],label=labels[j]) for j in range(len(labels))]
    legend+=[plt.scatter([],[],marker='X',c='gray',s=100,label='Out Range'),
             plt.Line2D([0],[0],color='red',lw=3,label='Interval Violation')]
    ax.legend(handles=legend,bbox_to_anchor=(1.05,1),fontsize=11,loc='upper left')
    
    plt.tight_layout()
    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=600,bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.post("/predict",summary="Input ≥2 SMILES, returns scoring plot and recommended method")
def predict(req:PredictRequest):
    """Main prediction endpoint."""
    smiles_list=req.smiles_list
    # Get predictions for all methods
    preds=[hub.predict(s)for s in smiles_list]
    
    # Organize predictions by method in fixed order
    datasets=[]
    for method in FIXED_METHOD_ORDER:
        method_predictions = []
        for p in preds:
            # Get prediction for this method, use None if method not available
            method_predictions.append(p.get(method, None))
        datasets.append(method_predictions)
    
    # Filter out methods with no valid predictions
    valid_datasets = []
    valid_methods = []
    for i, dataset in enumerate(datasets):
        # Check if all predictions in dataset are None
        if all(p is None for p in dataset):
            continue
        valid_datasets.append(dataset)
        valid_methods.append(FIXED_METHOD_ORDER[i])
    
    if not valid_datasets:
        raise HTTPException(status_code=400,detail="No valid predictions from any method")
    
    # Evaluate datasets
    results=eval_sys.evaluate_datasets(valid_datasets,valid_methods)
    
    # Find best method
    valid=[(r['method_name'],r['final_score'])for r in results if r['final_score']>=0]
    if not valid:
        raise HTTPException(status_code=400,detail="All methods penalized")
    
    recommended=max(valid,key=lambda x:x[1])[0]
    plot_b64=plot_to_b64(results)
    
    return {
        "plot_png_base64":plot_b64,
        "recommended_method":recommended,
        "available_methods":valid_methods
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "loaded_methods": list(hub.models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16828)