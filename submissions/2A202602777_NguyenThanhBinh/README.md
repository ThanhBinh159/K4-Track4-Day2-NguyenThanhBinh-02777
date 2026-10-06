# Chạy Lab Day 2 trên Kaggle

File import: **`code/lab_day2.ipynb`**. Notebook tự ghi đủ 9 module vào
`/kaggle/working/day2_code` và kiểm tra SHA256 trước khi chạy.
Không cần upload từng file Python. Các notebook `notebook6dba...` ở thư mục gốc là bản cũ.

Tài liệu yêu cầu gốc: [README](../README.md), [GUIDE](../GUIDE.md), [RUBRIC](../RUBRIC.md).
Ba file này, `eval.py` gốc và bộ khung `starter/` được giữ nguyên. Phần hoàn thiện nằm trong `code/`.

Link notebook đã lưu để nộp bài: **người chạy điền sau khi Save Version trên Kaggle**.

## 1. Chuẩn bị Dataset từ máy local

Giữ nguyên các CSV fold 0 của tác giả và ảnh. Không tự chia lại hay sửa nhãn.

```text
data/
  images/                         # 17.509 ảnh; hoặc cung cấp images.zip
    2016....jpg
  labels/
    labels.csv
    train_subset0.csv
    val_subset0.csv
    test_subset0.csv
```

1. Tạo Kaggle Dataset từ các file local bằng mục upload Dataset/Input của Kaggle.
   Có thể upload thư mục hoặc ZIP giữ cấu trúc trên, rồi xem cấu trúc đã giải nén trong Input.
2. Gắn Dataset vào notebook bằng **Add Input**. Nếu Dataset hiện tại đã có `data/images`
   và `data/labels`, dùng luôn Dataset đó.
3. Nếu dùng `images.zip`, notebook kiểm tra MD5 và giải nén xuống ổ đĩa; không giữ ảnh trong RAM.
4. Khi có nhiều Dataset đúng cấu trúc, điền một phần slug vào `DATASET_SLUG` để chọn rõ bộ dữ liệu.
   Bộ tìm dữ liệu hỗ trợ cả `/kaggle/input/<slug>` và
   `/kaggle/input/datasets/<owner>/<slug>`; không hardcode tên tài khoản.
5. `Species` là metadata tùy chọn. Các split chỉ cần `Filename,Label`; tên lớp mặc định
   lấy từ evaluator gốc nếu `labels.csv` không có `Species`.

## 2. Import và kiểm tra pipeline

1. Tạo notebook hoặc mở notebook cần chạy, dùng **File → Import Notebook**,
   chọn `code/lab_day2.ipynb` ở máy local.
2. Bật Internet để tải trọng số pretrained và thư viện. Chọn **GPU T4 ×2**.
   Notebook kiểm tra CUDA; khi chưa có GPU sẽ dừng trước khi huấn luyện.
3. Giữ `STAGE = "prepare"`; giữ cả `RESUME_OUTPUT_DIR` và `RESUME_ARCHIVE` là chuỗi rỗng
   ở lần chạy mới. Bấm **Run All**.
4. Kiểm tra log SHA256, GPU, số ảnh từng split/lớp, giao rỗng, hợp đủ 17.509 ảnh,
   so sánh với Table 1, ảnh mẫu, ảnh sau augmentation, loss ban đầu và overfit batch nhỏ.
5. Chỉ đi tiếp khi smoke test đạt `final_ce < 0.1`. Xem cả giá trị CE ban đầu so với `ln(9)`;
   pretrained head ngẫu nhiên có thể lệch mốc này nên notebook in số thực tế.

Đây là điểm dừng kiểm tra pipeline theo GUIDE §1.5. `prepare` không chạy B/T/F.
Mỗi lần **Run All** đều chạy các cell setup để khôi phục biến Python; không chạy thẳng cell B03
ngay sau khi import hoặc restart kernel.

## 3. Ưu tiên kiểm tra ConvNeXt

Sau khi `prepare` đạt, đặt:

```python
STAGE = "backbones"
RUN_BACKBONES_B01_B02 = False
RUN_BACKBONES_B03_B05 = True
BACKBONE_FILTER = ["B03"]
```

Bấm Run All. B03 chạy đầy đủ 12 epoch với công thức chung; kết quả này được dùng trong bảng backbone.
Không cần chạy B01/B02 trước để biết B03 có còn gặp vấn đề trên môi trường Kaggle hiện tại hay không.

Các thay đổi giới hạn tài nguyên:

- ConvNeXt mặc định **một T4**, AMP và channels-last. Chính sách áp dụng theo tên backbone,
  kể cả khi ConvNeXt được chọn cho các thí nghiệm T/F.
- B01/B02/B04 dùng `torch.nn.DataParallel` khi có hai GPU và `USE_DATA_PARALLEL=True`.
- B05 chạy một T4, tensor chuẩn thay vì channels-last: lần chạy DataParallel/channels-last
  đã dừng ngay batch đầu với `CUDA error: misaligned address`. Batch 32, 12 epoch,
  AMP và các tham số huấn luyện khác giữ nguyên. Chưa xác nhận riêng thành phần nào gây lỗi.
- Tất cả backbone cùng batch 32, 12 epoch, đầu vào 224, AdamW và warmup/cosine theo GUIDE.
  Batch giảm từ giá trị gợi ý 64 xuống 32 cho toàn bộ nhóm; không tự giảm riêng model nào khi OOM.
- Mỗi lần train là một tiến trình mới. Khi kết thúc, bộ nhớ model/optimizer/cache của tiến trình đó được thu hồi.
- Mặc định `num_workers=0`, CPU thread=1; ảnh giải mã từng batch, không cache toàn Dataset trong RAM.
- Giới hạn cache cuDNN trước khi import torch; không bật tìm thuật toán cuDNN cho mỗi shape.
- Không giữ thêm bản sao best model trên GPU, không giữ state resume trong RAM suốt quá trình train.
- Kiểm tra RAM mỗi 25 batch; dừng với `MemoryError` nếu RSS worker ≥18 GiB hoặc headroom
  cgroup (đã tính phần cache có thể thu hồi) <3 GiB. Đây là ngưỡng phòng ngừa, không bảo đảm
  chặn được mọi đợt tăng bộ nhớ đột ngột từ driver/hệ thống.
- Log văn bản ở batch đầu/cuối và tối đa mỗi phút; không có widget `train batches`.
  Các dòng ghi epoch, loss, thời gian, RSS/cgroup và VRAM.

CPU 200% có thể là hai lõi đang bận; số đó không tự chứng minh kernel chết vì CPU.
Giới hạn [cache cuDNN của PyTorch](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
và chi phí sao chép model trên từng forward của
[DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
là cơ sở của cấu hình thận trọng này. Nguyên nhân chính xác của lần chết cũ chưa được tái hiện trên T4.

Nếu B03 vẫn dừng, giữ nguyên `/kaggle/working/day2_output/runs/B03/seed0/run.log`,
`environment.json`, `history.csv` và log Kaggle. Không giảm epoch hoặc đổi split để che lỗi.
`B03_USE_DATA_PARALLEL=True` là tùy chọn thử nghiệm, không phải mặc định đã được xác minh ổn định.

## 4. Chạy sàng lọc đầy đủ trên validation

Đặt lại:

```python
STAGE = "screen"
RUN_BACKBONES_B01_B02 = True
RUN_BACKBONES_B03_B05 = True
BACKBONE_FILTER = []
```

Run All hoặc Save Version → Save & Run All cho phiên chạy dài. **Phiên Commit có thể là
runtime mới:** phải gắn output đã lưu bằng resume nếu muốn tái dùng B03/B01/B02 từ phiên trước.

Notebook chạy:

- B01 ResNet50, B02 ResNeXt50, B03 ConvNeXt-Tiny, B04 DeiT-Small, B05 EfficientNet-B0.
- T00 nền; T01 scratch; T02 frozen; T03 color jitter; T04 Mixup; T05 CutMix;
  T06 label smoothing; T07 focal; T08 class-weighted CE; T09 EMA; T10 kết hợp augmentation + loss.
- I00 một view; I01 lật và trung bình xác suất; I02 năm crop; I03 lật và trung bình logit;
  I08 AMP; I07 temperature scaling trên phương pháp chọn bằng val.
- Latency warmup ≥10, đo ≥50 lượt, đồng bộ GPU; batch 1 và 32; p50/p95/p99 và throughput.
  Có tính tạo/gộp view và calibration; không tính decode ảnh, resize/normalize nền hoặc truyền CPU→GPU.

Có thể chia nhiều phiên bằng `STAGE="backbones"`, `"training"`, `"inference"`.
Phải hoàn tất đủ năm backbone trước training; đủ T00–T10 trước inference.
Notebook ghi `results.xlsx` sau các stage. Không có metric test ở giai đoạn sàng lọc.

## 5. Tiếp tục từ checkpoint, không chạy lại B01/B02

Trong cùng runtime, chỉ bấm Run All với config cũ: mỗi run hoàn tất được tái dùng khi config,
source, split CSV và artifact đều khớp. Run dở tiếp tục từ **epoch đã lưu gần nhất**;
các batch của epoch bị ngắt sẽ chạy lại.

Ở runtime mới:

1. Lưu/download toàn bộ `day2_output` (đặc biệt `runs/*/seed*/checkpoint_last.pt` và `checkpoint_best.pt`).
2. Upload thư mục này thành Dataset và Add Input vào notebook.
3. Điền **đúng một** tùy chọn, bằng đường dẫn thực tế trong Input:

```python
RESUME_OUTPUT_DIR = "/kaggle/input/<output-dataset>/day2_output"
RESUME_ARCHIVE = ""
# Hoặc: RESUME_OUTPUT_DIR=""; RESUME_ARCHIVE="/kaggle/input/<output-dataset>/day2_output.zip"
```

4. Run All với các thông số huấn luyện giữ nguyên. Output cũ được copy sang working.
5. Muốn chỉ B03: dùng cấu hình mục 3. Muốn cả B03/B04/B05: `BACKBONE_FILTER=[]`,
   giữ nhóm B01/B02 False. Về sau phải giữ đủ artifact của B01/B02 để so sánh đủ năm backbone.

Nếu B01–B04 đã xong và B05 lỗi, khôi phục output cũ rồi đặt `STAGE="backbones"`,
`RUN_BACKBONES_B01_B02=False`, `RUN_BACKBONES_B03_B05=True`, `BACKBONE_FILTER=["B05"]`.
Run All sẽ chạy setup rồi chỉ train B05. Nếu B05 cũ chưa có checkpoint/kết quả và cấu hình
khác, notebook chuyển riêng thư mục `runs/B05/seed0` cũ sang tên `seed0_failed_dp_*`
để giữ log. Nếu đã có checkpoint/kết quả B05, notebook dừng để tránh mất tiến độ.
Một runtime Kaggle mới chỉ tái dùng B01–B04 khi đã gắn/khôi phục đủ `day2_output`;
output hiển thị trong file `.ipynb` không chứa checkpoint.

**Chỉ có `.ipynb` với output chữ/ảnh không đủ để resume.** Nó không chứa trọng số hoặc biến Python.
Không lấy các log cũ làm checkpoint. Bản triển khai này không tự nhận checkpoint của code cũ
khi config/source khác; cần một bộ output mới để kết quả nhất quán.

Không copy đè output resume cũ lên tiến độ mới mỗi lần chạy cell setup. Notebook chỉ restore
khi chưa có dữ liệu run trong working; xem thông báo restore/skip trong cell setup.

## 6. Chốt cấu hình và test

1. Duyệt workbook và đồ thị validation; ghi `SELECTION_REASON` dựa trên F1 và latency thực đo.
2. Đặt `STAGE="final"`, giữ `FINAL_SEEDS=[0,1,2]`, Run All.
3. Notebook ghi `final_selection.json` trước khi mở test, chạy T00/I00 và F01 với cùng ba seed.
   Checkpoint vẫn chọn bằng val; không gộp val vào train.
4. Test đã có CSV sẽ được đọc và kiểm tra lại, không forward lại để chọn điểm tốt hơn.
5. Sau final chỉ dùng `STAGE="final"` để tiếp tục phần còn thiếu hoặc `"report"` để xuất lại.
   Không quay lại sàng lọc cấu hình dựa trên test.

Khi dùng calibration, file `F01_uncal_seed*_test.csv` được ghi từ cùng lượt test trước hiệu chuẩn.
T fit từ val, không từ test. Workbook tính mean/std mẫu với `ddof=1` qua seed.

## 7. Tải và hoàn thiện bài nộp

Trong Output tải:

- `day2_submission.zip`: workbook, predictions, curves, log/config, hình phân bố lớp,
  tradeoff, confusion/error examples (sau final), evaluator và source Python.
- Notebook đã chạy có output: download `.ipynb` riêng, đưa vào `code/` của bài nộp.
- Toàn bộ `day2_output` nếu còn cần resume. ZIP sản phẩm nhỏ không chứa checkpoint.
  Có thể bật `PACKAGE_RESUME_OUTPUT=True` để tạo `day2_output.zip`; cần đủ dung lượng
  cho cả checkpoint và bản ZIP tạm thời.

Workbook gồm Backbones, Training, Inference, Final, PerClass, Latency, Summary.
`eval.py score` và `eval.py grade` được gọi sau final; kết quả ở thư mục `eval/`.
Mỗi run lưu `val_logits.npy`; vòng final lưu `test_logits.npy` từ chính lượt test đó.
Với TTA, logits có dạng K × N × 9; một view có dạng N × 9. `test_log_probs.npy`
là log xác suất sau gộp view, trước calibration, để đối chiếu riêng bước hiệu chuẩn.
GMAC là ước lượng Conv2d + Linear + QK/AV attention, bỏ qua norm/activation/pooling; bảng ghi rõ cách đếm.

Theo README gốc, bài nộp còn cần README riêng với link Kaggle và `report.md`/PDF.
Chép hướng dẫn này thành README riêng, điền link/version từ `runtime.json`, rồi hoàn thiện
báo cáo theo GUIDE §6.3 khi có kết quả thật. Không tự xem ZIP hiện tại là bài đã nộp hoàn chỉnh.

## 8. Kiểm tra local và đồng bộ khi sửa source

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m unittest discover -s code/tests -v
python code/sync_notebook.py
```

Dùng Python có torch, torchvision, timm 1.0.30, numpy, pandas, scikit-learn, matplotlib,
Pillow và openpyxl. Không thay torch/CUDA của Kaggle bằng bản CPU trên máy local.
`sync_notebook.py` cập nhật module nhúng/SHA và xóa output **trong notebook local mới**;
đừng chạy nó trên bản kết quả duy nhất chưa sao lưu.

Kiểm tra CPU không thay thế một lần chạy Kaggle T4. Bản phát hành chưa chứa số liệu thí nghiệm DeepWeeds.
