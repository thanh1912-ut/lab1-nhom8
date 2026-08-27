# Lab 1 — PyTorch FashionMNIST: Optimizer Benchmark (nhóm 8)

Benchmark **8 optimizer đơn × 3 learning rates × 3 seeds × 2 kiến trúc (MLP, CNN) = 144 runs** trên FashionMNIST để trả lời: **nên chọn optimizer nào?**

> Kết quả benchmark 10-epoch cũ (48 runs) nằm ở `archive/old_README.md`.

## Thiết kế thí nghiệm

### Dữ liệu
- FashionMNIST (28×28 grayscale, 10 lớp), transform `ToTensor()`
- **Train/Val/Test = 54k/6k/10k** — test chỉ dùng đánh giá best weight cuối cùng

### Kiến trúc
- **MLP**: Flatten → 784 → 256 → ReLU → 128 → ReLU → 10 (~235k params)
- **CNN**: Conv(1→16) → ReLU → MaxPool → Conv(16→32) → ReLU → MaxPool → FC 128 → 10 (~245k params)

### 8 optimizer đơn × 3 LR × 3 seed
| Optimizer | LR grid | Seed |
|-----------|---------|------|
| SGD | 0.05 / 0.01 / 0.001 | 42, 123, 2026 |
| Adadelta | 1.0 / 0.5 / 0.1 | 42, 123, 2026 |
| Adagrad | 0.01 / 0.005 / 0.001 | 42, 123, 2026 |
| RMSprop | 0.005 / 0.001 / 0.0005 | 42, 123, 2026 |
| Adam | 0.002 / 0.001 / 0.0005 | 42, 123, 2026 |
| AdamW | 0.002 / 0.001 / 0.0005 | 42, 123, 2026 |
| Adamax | 0.002 / 0.001 / 0.0005 | 42, 123, 2026 |
| NAdam | 0.002 / 0.001 / 0.0005 | 42, 123, 2026 |

### Chế độ training mỗi run
- Max **150 epochs**, batch 64, CrossEntropyLoss
- **Validate mỗi 5 epoch**: val loss + **acc / precision / recall / F1 (macro)** + **confusion matrix** (PNG lưu tại run dir, metrics ghi TensorBoard)
- **Early stop theo val loss**: 30 epochs không cải thiện → dừng
- **Best weight** theo val loss nhỏ nhất → cuối run đánh giá **test set đầy đủ** (acc/pre/rec/F1 + confusion matrix)
- **Log kiểu YOLO**: box header đầu run, 1 dòng mỗi epoch, bảng metrics khi validate
- **TensorBoard live**: `tensorboard --logdir outputs/tensorboard`
- **Checkpoint resume theo batch** (mỗi 200 batch + cuối mỗi epoch): model + optimizer + epoch/batch + RNG states → dừng bất cứ lúc nào (Ctrl+C / session Kaggle chết), chạy lại **resume đúng chỗ dừng**; run đã xong tự skip

### Git auto-backup (Kaggle)
Sau mỗi run xong, tự động commit & push lên GitHub: `results.json`, `train.log`, `cm_best.png` / `cm_test.png`, và `best.pt` của run tốt nhất mỗi (model, optimizer). Token đọc từ Kaggle Secrets `GITHUB_TOKEN`. → Session chết không mất dữ liệu (tối đa mất 200 batch train dở).

## Cấu trúc
```
├── configs/{mlp,cnn}.yaml    # mọi hyperparams chỉnh ở đây
├── src/
│   ├── data.py               # FashionMNIST + split 54k/6k/10k
│   ├── models.py             # MLP, CNN
│   ├── metrics.py            # acc/pre/rec/F1 + confusion matrix + vẽ CM
│   ├── logger.py             # YOLO-style console log + file log
│   ├── train.py              # train engine (early stop, resume, TensorBoard)
│   ├── benchmark.py          # quét yaml → 8 opt × 3 lr × 3 seed
│   └── gitbackup.py          # auto-push artifacts lên GitHub
├── train_mlp.py / train_cnn.py
├── kaggle_run.ipynb          # notebook chạy trên Kaggle
└── outputs/                  # runs/ (logs+weights+CM), tensorboard/, results.json
```

## Chạy

### Trên Kaggle (khuyến nghị)
1. Tạo PAT (scope `repo`) → thêm vào Kaggle Secrets tên `GITHUB_TOKEN`
2. Bật GPU T4 + Internet, upload/dùng notebook `kaggle_run.ipynb`
3. Chạy tuần tự `train_mlp.py` → `train_cnn.py`. Xem TensorBoard ngay trong notebook. Session chết → chạy lại notebook, tự resume.

### Local
```bash
pip install -r requirements.txt
python train_mlp.py            # hoặc --only adam để chạy 1 optimizer
python train_cnn.py
tensorboard --logdir outputs/tensorboard
tail -f outputs/runs/mlp_adam_lr0.001_s42/train.log   # xem log live
```

## Kết quả

*(khoảng này sẽ cập nhật sau khi train xong 144 runs — script tổng hợp + plots sẽ được thêm)*

## Phân công nhóm & yêu cầu lab mapping

| Yêu cầu lab | Đáp ứng |
|---|---|
| Load FashionMNIST + transforms | `src/data.py` |
| Build neural network | `src/models.py` (MLP + CNN) |
| Training loop (forward/loss/backward/optimize) | `src/train.py` |
| Evaluate accuracy | `src/metrics.py` (+pre/rec/F1, CM) |
| Save/load model | best.pt/last.pt + verify khi test |
| Experiment hyperparams | 8 opt × 3 lr × 3 seed × 2 model |
| Visualize loss | TensorBoard + train.log |
| Predicted vs actual images | `cm_test.png` + (sẽ thêm grid ảnh) |
