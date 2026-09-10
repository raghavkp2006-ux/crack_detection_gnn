import os, sys, glob, json, cv2, base64, time, torch
from typing import Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from sklearn.metrics import f1_score, precision_score, recall_score
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DCS_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DEEPSEG_DIR = os.path.join(DCS_ROOT, 'DeepSegmentor')
if DEEPSEG_DIR not in sys.path:
    sys.path.insert(0, DEEPSEG_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.hybrid_model import build_model
from src.train import load_config
from src.utils.dataset import CrackGraphDataset
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data

# Failsafe simulation check
ALLOW_SIMULATION = os.getenv("DCS_ALLOW_SIMULATION", "false").lower() == "true"

class ModelNotAvailableError(Exception):
    pass

app = FastAPI(title="Pipeline Crack GNN & DeepCrack Interactive Server", version="2.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')
TEST_IMG_DIR = os.path.abspath(os.path.join(DCS_ROOT, 'DeepCrack', 'test_img'))
TEST_LAB_DIR = os.path.abspath(os.path.join(DCS_ROOT, 'DeepCrack', 'test_lab'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Starting GNN & DeepCrack Engine on device: {device}')

# Model registry
MODELS = {}
CONFIGS = {
    'M1': ('configs/baseline_deep_gnn.yaml', 'results/M1_deep_gnn/best_model.pt', 0.60, 'Deep GNN (8 Layers)'),
    'M2': ('configs/shallow_gnn.yaml', 'results/M2_shallow_gnn/best_model.pt', 0.70, 'Shallow GNN (2 Layers)'),
    'M3': ('configs/hybrid_no_pe.yaml', 'results/M3_hybrid_no_pe/best_model.pt', 0.74, 'Hybrid GNN+Transformer (No PE)'),
    'M4': ('configs/step2_loss/hybrid_full_focal.yaml', 'results/step2_loss/M4_hybrid_full/best_model.pt', 0.66, 'Hybrid GNN+Transformer (Full PE)'),
}

for key, (cfg_path, ckpt_path, default_tau, desc) in CONFIGS.items():
    full_ckpt = os.path.join(BASE_DIR, ckpt_path)
    if os.path.exists(full_ckpt):
        cfg = load_config(os.path.join(BASE_DIR, cfg_path))
        model = build_model(cfg).to(device)
        ckpt = torch.load(full_ckpt, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        MODELS[key] = {
            'model': model,
            'tau': default_tau,
            'desc': desc,
            'type': cfg['model']['type']
        }
        print(f'Loaded model [{key}]: {desc} (tau={default_tau})')
    else:
        print(f'Warning: Checkpoint for {key} not found at {full_ckpt}')

# Load DeepCrack if trained
dc_ckpt = os.path.join(BASE_DIR, 'results', 'multiseed', 'deepcrack', 'seed_42', 'best_model.pt')
if not os.path.exists(dc_ckpt):
    # Try any available seed
    dc_ckpts = glob.glob(os.path.join(BASE_DIR, 'results', 'multiseed', 'deepcrack', 'seed_*', 'best_model.pt'))
    if dc_ckpts:
        dc_ckpt = dc_ckpts[0]

if os.path.exists(dc_ckpt):
    try:
        from models.deepcrack_networks import DeepCrackNet
        dc_net = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch').to(device)
        dc_net.load_state_dict(torch.load(dc_ckpt, map_location=device))
        dc_net.eval()
        MODELS['DeepCrack'] = {
            'model': dc_net,
            'tau': 0.36,
            'desc': 'DeepCrack (Liu et al. 2019, Pixel CNN)',
            'type': 'deepcrack'
        }
        print(f'Loaded model [DeepCrack]: DeepCrack (Liu et al. 2019) (tau=0.36)')
    except Exception as e:
        print(f'Error loading DeepCrackNet: {e}')

# Preload holdout test graphs for instant retrieval
test_graph_dataset = CrackGraphDataset(os.path.join(BASE_DIR, 'data/graphs/test'))
test_graph_files = sorted(glob.glob(os.path.join(BASE_DIR, 'data/graphs/test/*.pt')))
test_graph_map = {}
for i, fpath in enumerate(test_graph_files):
    img_id = os.path.splitext(os.path.basename(fpath))[0]
    test_graph_map[img_id] = i

print(f'Indexed {len(test_graph_map)} holdout test graphs for instant response.')


@app.get('/api/models')
def get_models():
    return {
        key: {
            'description': info['desc'],
            'optimal_threshold': info['tau'],
            'type': info['type']
        } for key, info in MODELS.items()
    }


@app.get('/api/telemetry')
def get_telemetry():
    pred_path = os.path.join(DASHBOARD_DIR, 'predictions_data.json')
    if not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail="predictions_data.json not found")
    with open(pred_path, 'r') as f:
        data = json.load(f)
    return data


def run_model_on_data(key, data, tau=None):
    if key not in MODELS:
        if not ALLOW_SIMULATION:
            raise ModelNotAvailableError(f"Model '{key}' is not available.")
        # Simulation placeholder
        preds = np.random.binomial(1, 0.05, size=data.num_nodes)
        return preds.astype(float), preds, 0.50

    minfo = MODELS[key]
    model = minfo['model']
    threshold = tau if tau is not None else minfo['tau']

    with torch.no_grad():
        if minfo['type'] == 'hybrid':
            logits = model(data.x, data.edge_index, data.pos)
        else:
            logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= threshold).astype(int)
    return probs, preds, threshold


def run_deepcrack_on_image(img_rgb, tau=None):
    if 'DeepCrack' not in MODELS:
        if not ALLOW_SIMULATION:
            raise ModelNotAvailableError("Model 'DeepCrack' is not available.")
        H, W = img_rgb.shape[:2]
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.int32), 0.50

    minfo = MODELS['DeepCrack']
    model = minfo['model']
    threshold = tau if tau is not None else minfo['tau']

    H, W = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, _, _, fused = model(tensor)
        fused_up = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=True)
        prob_map = torch.sigmoid(fused_up)[0, 0].cpu().numpy()
        binary_mask = (prob_map >= threshold).astype(np.int32)

    return prob_map, binary_mask, threshold


def render_overlay(img_rgb, segments, node_preds, gt_mask=None, pixel_mask=None):
    H, W = img_rgb.shape[:2]
    
    if pixel_mask is not None:
        # Pixel-level overlay for DeepCrack or direct mask
        marked = img_rgb.copy()
        red_mask = np.zeros_like(img_rgb)
        red_mask[pixel_mask == 1] = [239, 68, 68]
        has_defect = (pixel_mask == 1)
        if has_defect.any():
            marked[has_defect] = cv2.addWeighted(marked[has_defect], 0.45, red_mask[has_defect], 0.55, 0)
        # Find and draw contours for sharp boundary definition
        contours, _ = cv2.findContours((pixel_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, (56, 189, 248), 1)
        overlay = marked
    else:
        # Superpixel boundary visualization
        if segments.shape != (H, W):
            segments = cv2.resize(segments, (W, H), interpolation=cv2.INTER_NEAREST)
        marked = mark_boundaries(img_rgb, segments, color=(0.2, 0.7, 1.0))
        overlay = (marked * 255).astype(np.uint8)

        labels = np.unique(segments)
        red_mask = np.zeros_like(img_rgb)
        for i, label in enumerate(labels):
            if i < len(node_preds) and node_preds[i] == 1:
                seg_m = (segments == label)
                red_mask[seg_m] = [239, 68, 68]

        has_defect = red_mask.any(axis=-1)
        if has_defect.any():
            overlay[has_defect] = cv2.addWeighted(overlay[has_defect], 0.45, red_mask[has_defect], 0.55, 0)

    # Encode to base64
    _, buf_pred = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_pred = base64.b64encode(buf_pred).decode('utf-8')

    b64_gt = None
    if gt_mask is not None:
        gt_vis = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)
        _, buf_gt = cv2.imencode('.png', gt_vis)
        b64_gt = base64.b64encode(buf_gt).decode('utf-8')

    _, buf_orig = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_orig = base64.b64encode(buf_orig).decode('utf-8')

    return b64_orig, b64_pred, b64_gt


@app.get('/api/inspect/{image_id}')
def inspect_test_image(image_id: str, model: str = 'M4', tau: float = None):
    if image_id not in test_graph_map:
        raise HTTPException(status_code=404, detail=f'Image ID {image_id} not found in holdout dataset')

    idx = test_graph_map[image_id]
    data = test_graph_dataset[idx].to(device)

    img_path = os.path.join(TEST_IMG_DIR, f'{image_id}.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(TEST_IMG_DIR, f'{image_id}.png')
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f'Raw image for {image_id} not found')

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # Ground truth mask
    lab_path = os.path.join(TEST_LAB_DIR, f'{image_id}.png')
    gt_mask = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(lab_path) else None

    # Calculate morphology
    cat, ar, area_frac = 'Unknown', 0.0, 0.0
    if gt_mask is not None:
        pts = np.argwhere(gt_mask > 127)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            ar = float(max(w, h) / (min(w, h) + 1e-5))
            area_frac = float(len(pts) / (H * W))
            if area_frac < 0.025: cat = 'Thin_Fine_Fissure'
            elif ar >= 2.0: cat = 'Long_Elongated'
            else: cat = 'Branched_Complex'
        else:
            cat = 'Clean_Negative'

    # Compute superpixels
    segments = compute_superpixels(img_rgb, n_segments=300, compactness=10.0, sigma=1.0)

    # Run models for comparison
    results = {}
    y_true = data.y.cpu().numpy()
    active_preds = None
    active_pixel_mask = None

    for m_key in ['M1', 'M2', 'M3', 'M4']:
        if m_key in MODELS:
            custom_tau = tau if m_key == model else None
            probs, preds, used_tau = run_model_on_data(m_key, data, custom_tau)
            f1 = float(f1_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            prec = float(precision_score(y_true, preds, zero_division=0))
            results[m_key] = {
                'f1': round(f1, 4),
                'recall': round(rec, 4),
                'precision': round(prec, 4),
                'threshold': round(used_tau, 2),
                'defect_nodes': int((preds == 1).sum()),
                'total_nodes': len(preds)
            }
            if m_key == model:
                active_preds = preds

    # Also evaluate DeepCrack if available
    if 'DeepCrack' in MODELS:
        custom_tau = tau if model == 'DeepCrack' else None
        dc_prob_map, dc_pix_mask, dc_used_tau = run_deepcrack_on_image(img_rgb, custom_tau)
        
        # Pool to superpixel nodes for graph metrics
        labels = np.unique(segments)
        dc_node_probs = np.zeros(len(labels), dtype=np.float32)
        for s_i, seg_id in enumerate(labels):
            m = (segments == seg_id)
            if m.any(): dc_node_probs[s_i] = dc_prob_map[m].mean()
        dc_node_preds = (dc_node_probs >= dc_used_tau).astype(int)

        dc_f1 = float(f1_score(y_true, dc_node_preds, zero_division=0))
        dc_rec = float(recall_score(y_true, dc_node_preds, zero_division=0))
        dc_prec = float(precision_score(y_true, dc_node_preds, zero_division=0))
        results['DeepCrack'] = {
            'f1': round(dc_f1, 4),
            'recall': round(dc_rec, 4),
            'precision': round(dc_prec, 4),
            'threshold': round(dc_used_tau, 2),
            'defect_nodes': int((dc_node_preds == 1).sum()),
            'total_nodes': len(dc_node_preds)
        }
        if model == 'DeepCrack':
            active_pixel_mask = dc_pix_mask

    recommended_model = 'M4' if cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else 'M1'
    b64_orig, b64_pred, b64_gt = render_overlay(img_rgb, segments, active_preds, gt_mask, pixel_mask=active_pixel_mask)

    return {
        'id': image_id,
        'category': cat,
        'aspect_ratio': round(ar, 2),
        'area_fraction': round(area_frac, 4),
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'ground_truth_crack_nodes': int((y_true == 1).sum()),
        'selected_model': model,
        'router_recommendation': recommended_model,
        'model_comparison': results,
        'images': {
            'original': f'data:image/jpeg;base64,{b64_orig}',
            'prediction_overlay': f'data:image/jpeg;base64,{b64_pred}',
            'ground_truth': f'data:image/png;base64,{b64_gt}' if b64_gt else None
        }
    }


# Live segmentation endpoint (supporting both POST /segment-crack and POST /api/predict_upload)
@app.post('/segment-crack')
@app.post('/api/segment-crack')
@app.post('/api/predict_upload')
async def segment_crack_endpoint(
    file: UploadFile = File(...),
    model: str = Form('M4'),
    tau: Optional[float] = Form(None)
):
    # 1. Input Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in upload.")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")

    H, W = img_bgr.shape[:2]
    if H < 32 or W < 32 or H > 4096 or W > 4096:
        raise HTTPException(status_code=400, detail=f"Image dimensions ({W}x{H}) out of allowed bounds (32px to 4096px).")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Check model availability
    valid_models = ['M1', 'M2', 'M3', 'M4', 'DeepCrack']
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Valid options: {valid_models}")

    if model not in MODELS and not ALLOW_SIMULATION:
        raise HTTPException(
            status_code=503,
            detail=f"ModelNotAvailableError: Model checkpoint for '{model}' is not available on the server and DCS_ALLOW_SIMULATION is false."
        )

    t0 = time.time()
    results = {}
    active_preds = None
    active_pix_mask = None
    area_frac, ar, cat = 0.0, 0.0, 'Clean_Negative'

    # 3. Execution Pipeline
    if model == 'DeepCrack':
        t_infer_start = time.time()
        prob_map, pix_mask, used_tau = run_deepcrack_on_image(img_rgb, tau)
        t_infer = time.time() - t_infer_start
        t_prep = t_infer_start - t0
        active_pix_mask = pix_mask

        pts = np.argwhere(pix_mask > 0)
        num_defects = len(pts)
        total_defects = H * W
        area_frac = float(num_defects / (H * W))
        if num_defects > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            ar = float(max(w, h) / (min(w, h) + 1e-5))
            if area_frac < 0.025: cat = 'Thin_Fine_Fissure'
            elif ar >= 2.0: cat = 'Long_Elongated'
            else: cat = 'Branched_Complex'

        results['DeepCrack'] = {
            'threshold': round(used_tau, 2),
            'defect_pixels': num_defects,
            'defect_fraction': round(area_frac, 4),
            'total_pixels': total_defects
        }
        segments = np.zeros((H, W), dtype=int)
        b64_orig, b64_pred, _ = render_overlay(img_rgb, segments, None, pixel_mask=active_pix_mask)

    else:
        # GNN Pipeline (M1, M2, M3, M4)
        t_prep_start = time.time()
        segments = compute_superpixels(img_rgb, n_segments=300, compactness=10.0, sigma=1.0)
        feats, centroids = extract_node_features(img_rgb, segments)
        edge_index, edge_weight = build_pirm_graph(feats, k=8, sigma=0.1)
        edge_index, edge_weight = symmetrize_graph(edge_index, edge_weight)
        data = build_pyg_data(feats, edge_index, edge_weight, centroids).to(device)
        t_prep = time.time() - t_prep_start

        t_infer_start = time.time()
        for m_key in ['M1', 'M2', 'M3', 'M4']:
            if m_key in MODELS:
                custom_tau = tau if m_key == model else None
                probs, preds, used_tau = run_model_on_data(m_key, data, custom_tau)
                results[m_key] = {
                    'threshold': round(used_tau, 2),
                    'defect_nodes': int((preds == 1).sum()),
                    'defect_fraction': round(float((preds == 1).sum() / len(preds)), 4),
                    'total_nodes': len(preds)
                }
                if m_key == model:
                    active_preds = preds

        t_infer = time.time() - t_infer_start

        # Calculate morphology from superpixel predictions
        crack_pixel_mask = np.zeros((H, W), dtype=np.uint8)
        labels = np.unique(segments)
        for i, seg_id in enumerate(labels):
            if i < len(active_preds) and active_preds[i] == 1:
                crack_pixel_mask[segments == seg_id] = 1

        pts = np.argwhere(crack_pixel_mask > 0)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            ar = float(max(w, h) / (min(w, h) + 1e-5))
            area_frac = float(len(pts) / (H * W))
            if area_frac < 0.025: cat = 'Thin_Fine_Fissure'
            elif ar >= 2.0: cat = 'Long_Elongated'
            else: cat = 'Branched_Complex'

        b64_orig, b64_pred, _ = render_overlay(img_rgb, segments, active_preds)

    total_time = time.time() - t0
    crack_detected = area_frac > 0.0005

    return {
        'status': 'success',
        'filename': file.filename,
        'model': model,
        'crack_detected': crack_detected,
        'image_dimensions': [H, W],
        'metrics': {
            'crack_area_fraction': round(area_frac, 4),
            'aspect_ratio': round(ar, 2),
            'morphology_bucket': cat,
            'defect_elements': int((active_preds == 1).sum()) if active_preds is not None else int((active_pix_mask == 1).sum()),
            'total_elements': len(active_preds) if active_preds is not None else (H * W)
        },
        'timing': {
            'preprocessing_sec': round(t_prep, 3),
            'inference_sec': round(t_infer, 4),
            'total_sec': round(total_time, 3)
        },
        'results': results,
        'images': {
            'original': f'data:image/jpeg;base64,{b64_orig}',
            'prediction_overlay': f'data:image/jpeg;base64,{b64_pred}'
        }
    }


# Mount dashboard static files
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

@app.get('/', response_class=HTMLResponse)
def root():
    index_file = os.path.join(DASHBOARD_DIR, 'index.html')
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            return f.read()
    return '<h1>Pipeline Crack GNN Server is active. Navigate to <a href="/dashboard">/dashboard</a></h1>'

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
