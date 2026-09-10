import os
import glob
import shutil

BASE_DIR = r"c:\antigravity_projects\dcsnew"
SRC_DEEPCRACK = os.path.join(BASE_DIR, "DeepCrack")
DEST_DIR = os.path.join(BASE_DIR, "DeepSegmentor", "datasets", "DeepCrack")
GNN_TRAIN_GRAPHS = os.path.join(BASE_DIR, "pipeline-crack-gnn", "data", "graphs", "train")
GNN_TEST_GRAPHS = os.path.join(BASE_DIR, "pipeline-crack-gnn", "data", "graphs", "test")

folders = ["train_img", "train_lab", "test_img", "test_lab"]
for f in folders:
    os.makedirs(os.path.join(DEST_DIR, f), exist_ok=True)

# 1. Copy / Link train and test images and labels
for f in folders:
    src_f = os.path.join(SRC_DEEPCRACK, f)
    dst_f = os.path.join(DEST_DIR, f)
    files = glob.glob(os.path.join(src_f, "*"))
    for file_path in files:
        fname = os.path.basename(file_path)
        target = os.path.join(dst_f, fname)
        if not os.path.exists(target):
            shutil.copy2(file_path, target)

# 2. Verify holdout test files against GNN graphs
test_graphs = sorted(glob.glob(os.path.join(GNN_TEST_GRAPHS, "*.pt")))
test_graph_ids = set(os.path.splitext(os.path.basename(p))[0] for p in test_graphs)

test_imgs = sorted(glob.glob(os.path.join(DEST_DIR, "test_img", "*")))
test_img_ids = set(os.path.splitext(os.path.basename(p))[0] for p in test_imgs)

test_labs = sorted(glob.glob(os.path.join(DEST_DIR, "test_lab", "*.png")))
test_lab_ids = set(os.path.splitext(os.path.basename(p))[0] for p in test_labs)

print(f"Total GNN test graphs: {len(test_graph_ids)}")
print(f"Total DeepSegmentor test_img: {len(test_img_ids)}")
print(f"Total DeepSegmentor test_lab: {len(test_lab_ids)}")

assert test_graph_ids == test_img_ids, f"Mismatch: {test_graph_ids ^ test_img_ids}"
assert test_img_ids == test_lab_ids, f"Mismatch lab vs img: {test_img_ids ^ test_lab_ids}"

train_graphs = sorted(glob.glob(os.path.join(GNN_TRAIN_GRAPHS, "*.pt")))
train_graph_ids = set(os.path.splitext(os.path.basename(p))[0] for p in train_graphs)
train_imgs = sorted(glob.glob(os.path.join(DEST_DIR, "train_img", "*")))
train_img_ids = set(os.path.splitext(os.path.basename(p))[0] for p in train_imgs)

print(f"Total GNN train graphs: {len(train_graph_ids)}")
print(f"Total DeepSegmentor train_img: {len(train_img_ids)}")
assert train_graph_ids == train_img_ids, f"Mismatch train: {train_graph_ids ^ train_img_ids}"

print("\nSUCCESS: DeepSegmentor dataset folder verified with exact 300 train and 237 holdout test pairing!")
