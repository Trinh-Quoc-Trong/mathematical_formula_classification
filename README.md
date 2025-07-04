#target: phân lớp 3 kỹ hiệu toán học

<p align="center">
  <img src="https://e7.pngegg.com/pngimages/99/619/png-clipart-square-root-computer-icons-area-angle-others-angle-text-thumbnail.png" width="200"/>
  <img src="https://media.istockphoto.com/id/1175711636/vi/vec-to/d%E1%BA%A5u-hi%E1%BB%87u-sigma-%C4%91en.jpg?s=1024x1024&w=is&k=20&c=L6Lhz94DHMQNrvg1xf8dJFRwSDPUaZQzkWAd8YNp0sE=" width="200"/>
  <img src="https://m.media-amazon.com/images/I/51IQAsVSZmL.jpg" width="200"/>
</p>



Dưới đây là một “khung xương” (skeleton) 

• Tách biệt rõ dữ liệu – mã nguồn – sản phẩm.  
• Thuận tiện cho tái hiện (reproducibility), chia sẻ, CI/CD, và mở rộng.

```
project_name/
│
├── README.md               # Giới thiệu, hướng dẫn cài đặt/chạy nhanh
├── pyproject.toml | setup.py | requirements.txt
├── .env.example            # Biến môi trường nhạy cảm (API key, paths…)
├── .gitignore
├── data/                   # KHÔNG commit file lớn; dùng DVC, Git-LFS, S3…
│   ├── raw/                # Dữ liệu gốc (read-only)
│   ├── interim/            # Dữ liệu sau các bước xử lý trung gian
│   ├── processed/          # Dữ liệu đã sẵn sàng train/eval
│   └── external/           # Nguồn ngoài (pre-trained, benchmark…)
│
├── notebooks/              # EDA, prototype nhanh (đặt tên 01_, 02_…)
│
├── src/                    # Mã nguồn “production-ready”
│   ├── __init__.py
│   ├── config/             # YAML/JSON/Ωmegaconf hydra configs
│   ├── data/               # Load, transform, augment
│   │   └── make_dataset.py
│   ├── features/           # Trích xuất đặc trưng
│   ├── models/             # Định nghĩa model – sklearn, PyTorch Lightning…
│   ├── train.py            # Điểm vào huấn luyện
│   ├── evaluate.py         # Tính metric, vẽ curve
│   └── predict.py          # Inference lô hoặc realtime
│
├── scripts/                # Bash/Powershell cho automation (train.sh, deploy.sh…)
│
├── tests/                  # Unit & integration tests (pytest)
│
├── models/                 # Checkpoints, model card, ONNX/TorchScript
│
├── reports/                # Báo cáo tự động (HTML, PNG, PDF) – created by evaluate
│   └── figures/
│
├── docs/                   # Tài liệu (Sphinx, MkDocs) + kiến trúc hệ thống
│
└── ci/                     # Workflow: GitHub Actions / GitLab CI / Jenkinsfile
```

Giải thích ngắn gọn:

1. data/: chia 3–4 cấp cho lifecycle dữ liệu. Sử dụng DVC hoặc MLflow artifacts để quản lý version và tránh phình repo git.

2. src/:  
   • Tối ưu hoá tái dùng bằng cách đóng gói thành package (`python -m pip install -e .`).  
   • Chia “data → features → models” đúng trách nhiệm (single-responsibility).  
   • Cấu hình tách khỏi code (Hydra/OmegaConf) để tái lặp thí nghiệm dễ dàng: `python train.py model=resnet50 data.batch_size=64`.

3. notebooks/: chỉ để khám phá; kết quả quan trọng nên chuyển thành script trong src/.

4. tests/: mọi hàm tiền xử lý, loss, metric… nên có test; CI sẽ chạy `pytest` kèm `ruff`/`flake8` + `mypy`.

5. models/, reports/: sinh ra bởi pipeline, không commit file nặng; push lên object storage.

6. scripts/ + ci/: tự động hoá từ huấn luyện tới triển khai (build Docker, push model registry, deploy FastAPI).

7. docs/: viết kiến trúc, API reference, quyết định thiết kế để member mới “onboard” nhanh.

Mẹo thực chiến

• Dùng `Cookiecutter Data Science` hoặc `hydra-template` làm nền, tiết kiệm thời gian scaffold.  
• Chuẩn hoá logging (Python `logging`, `rich`, hoặc `loguru`).  
• Theo dõi thí nghiệm bằng MLflow, Weights & Biases.  
• Thiết lập pre-commit hook: ruff/black/isort, detect-large-files, nbstripout.  
• Nếu nhiều model, áp dụng monorepo + Poetry multiproject hoặc Nx (TS/JS).  
• Với sản phẩm online, tách repo inference (microservice) ra, hoặc dùng tritonserver.

Áp dụng cho dự án của bạn

Dự án “mathematical_formula_classification” hiện có khá nhiều dữ liệu .inkml và script thử nghiệm rải rác. Bạn có thể:

1. Di chuyển code xử lý .inkml vào `src/data/parse_inkml.py`.  
2. Viết `src/train.py` gom pipeline đọc processed pickle → train CNN/LSTM.  
3. Quản lý dữ liệu raw/processed qua DVC để đồng bộ với teammate.  
4. Viết `tests/test_parse_inkml.py` đảm bảo parser không hỏng khi nhập file mới.  
5. Dùng GitHub Actions: `pytest`, `flake8`, `dvc pull` trước khi train CI.

Với khung ở trên, việc mở rộng (thử model khác, thêm augmentation, deploy API nhận ảnh latex) sẽ nhanh gọn và đáng tin cậy hơn. Chúc bạn tổ chức repo thật gọn gàng và chuyên nghiệp!