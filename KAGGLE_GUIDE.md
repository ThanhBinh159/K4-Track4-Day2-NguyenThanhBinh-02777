# Hướng dẫn chạy `lab_day2.ipynb` trên Kaggle

> Bản hướng dẫn hiện hành cho notebook trong `code/` là [code/README.md](code/README.md).
> Phần dưới ghi lại quy trình của notebook cũ; không dùng đường dẫn `starter/lab_day2.ipynb`
> hoặc chỉ dẫn chạy B05 trên hai GPU cho bản hiện hành.

Tài liệu này hướng dẫn đưa notebook lên Kaggle, đóng gói dữ liệu local thành Kaggle Dataset, chạy các thí nghiệm có theo dõi tiến độ và tải kết quả về máy. Luồng thí nghiệm dùng fold 0; validation dùng để chọn cấu hình. Cell chung kết mặc định chưa chạy test.

Tài liệu tham khảo chính thức: [Kaggle Datasets](https://www.kaggle.com/docs/datasets) và [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks). Tên nút có thể thay đổi theo giao diện Kaggle.

## 1. Chuẩn bị file ở máy local

Notebook cần đúng các CSV fold 0 của bộ DeepWeeds và ảnh có tên trùng cột `Filename`:

```text
deepweeds-kaggle/
├── labels/
│   ├── labels.csv
│   ├── train_subset0.csv
│   ├── val_subset0.csv
│   └── test_subset0.csv
└── images.zip                 # hoặc thư mục images/ chứa các file JPG
```

Có thể upload `images.zip` hoặc các ảnh đã giải nén. Notebook tự tìm CSV trong `/kaggle/input`, xác nhận split có đủ 17.509 ảnh, rồi liên kết/copy ảnh vào thư mục làm việc để đọc nhanh hơn. Nếu archive là `images.zip` nguyên bản từ DeepWeeds, notebook in MD5 `b7b30f96d466fba86016aa5a26606e0f`; dù archive được đóng gói lại, nó vẫn kiểm tra tên ảnh theo CSV.

Không đổi nội dung các CSV và không tạo split mới. Chỉ cần `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv` và `labels.csv`; notebook không dùng fold khác.

## 2. Tạo Kaggle Dataset từ dữ liệu local

1. Đăng nhập Kaggle và mở mục **Datasets**, chọn tạo Dataset mới.
2. Đặt tên dễ nhận diện, chọn quyền riêng tư **Private** nếu dữ liệu cần giới hạn chia sẻ.
3. Kéo thả thư mục `deepweeds-kaggle/` hoặc chọn các file bên trong. Giữ cấu trúc `labels/` nếu đang dùng cấu trúc ở trên.
4. Hoàn tất tạo Dataset và chờ Kaggle xử lý upload.

Nếu giao diện Kaggle hỏi giấy phép cho Dataset, chọn theo nguồn dữ liệu DeepWeeds và chính sách lớp học của bạn. Không đưa token/API key vào notebook hoặc Dataset.

## 3. Import notebook và gắn Dataset

1. Mở [Kaggle Notebooks](https://www.kaggle.com/code) và tạo notebook mới.
2. Dùng chức năng **Import Notebook** hoặc mục import/upload notebook trong trình soạn thảo để chọn file `starter/lab_day2.ipynb` trên máy. Nếu Kaggle mở notebook trống, dùng menu import của editor.
3. Trong phần **Input**, chọn **Add Input**, tìm Dataset vừa tạo và gắn vào notebook.
4. Mở **Settings → Accelerator**, chọn **GPU T4 ×2**. Cấu hình notebook bật `USE_DATA_PARALLEL = True` cho các thí nghiệm khác; riêng B03 đặt `B03_USE_DATA_PARALLEL = False` để thử ConvNeXt-Tiny trên một T4 sau các lần kernel chết.
5. Bật **Internet** để cài `timm` khi runtime chưa có và tải pretrained weights. Kaggle có thể yêu cầu lưu notebook trước khi bật một số tùy chọn.

Kaggle gắn Dataset ở `/kaggle/input/`; thư mục này chỉ đọc. Notebook ghi checkpoint và kết quả vào `/kaggle/working/day2_output/`.

Nếu notebook thấy đúng một thư mục `labels/`, không cần sửa đường dẫn. Nếu gắn nhiều Dataset có cùng tên CSV, sửa `DATASET_SLUG` trong ô **Cấu hình runtime** thành một phần tên/slug của Dataset cần dùng, rồi chạy lại các ô từ cấu hình và nhận diện dữ liệu.

## 4. Chạy và theo dõi notebook

Trong ô **Cấu hình runtime**, để `RESUME_OUTPUT_DIR = ""` và `RESUME_ARCHIVE = ""` khi chạy lần đầu. Hai biến này chỉ dùng để nạp output của phiên trước đã upload thành Kaggle Dataset và gắn vào notebook; đường dẫn phải là đường dẫn tuyệt đối bên trong `/kaggle/input/`. Không điền `/kaggle/working/day2_output` vì đó là thư mục đích của phiên hiện tại, chưa có trước khi chạy cell.

Chạy các cell từ trên xuống để giữ trạng thái biến trong kernel. Trước khi thí nghiệm dài, kiểm tra cell đầu in `device=cuda`, `GPU count=2`, `CPU threads=1 | DataLoader workers=0` và `B03 DataParallel=False`. Các thí nghiệm khác in `DataParallel: đang dùng 2 GPU`; B03 in `DataParallel=tắt` và dùng GPU đầu tiên. Chế độ nạp ảnh bằng tiến trình chính giảm số tiến trình CPU và bộ nhớ đệm nhưng có thể khiến GPU chờ dữ liệu lâu hơn.

Các cell chính chạy theo thứ tự:

1. **Cài môi trường, đồng bộ code:** cài những thư viện thiếu, ghi các module Python vào `/kaggle/working/day2_code/`, rồi kiểm tra SHA256 để xác nhận chúng khớp với source đã nhúng trong notebook.
2. **Nhận diện dữ liệu và EDA:** tự tìm `labels/`, giải nén hoặc lập chỉ mục ảnh, kiểm tra split fold 0, in số ảnh theo lớp và hiển thị tối thiểu ba ảnh train mỗi lớp.
3. **Smoke test:** đo cross-entropy ban đầu, thử overfit một batch nhỏ và xem ảnh sau augmentation. Nếu cell báo lỗi hoặc loss không giảm, dừng ở đây để xử lý trước khi train.
4. **Backbone nhóm 1:** cell B01–B02 chạy ResNet-50 và ResNeXt-50. Nếu `experiment_records.json` và đủ checkpoint, biểu đồ, dự đoán validation, latency đã có, cell in `Reuse` và không train lại.
5. **Backbone nhóm 2:** cell B03–B05 chạy ConvNeXt-Tiny, DeiT-Small và EfficientNet-B0. B03 hiện thử trên một GPU; B04–B05 vẫn dùng hai GPU. Có thể chạy riêng cell này sau các cell setup dữ liệu và helper backbone.
6. **Training ablation:** so sánh T00 với frozen backbone, color augmentation, label smoothing, focal loss và T05 kết hợp các lựa chọn thắng trên validation.
7. **Inference/latency:** so sánh một view, flip TTA, năm crop, gộp logit và AMP. Đo latency có 10 warmup, 50 lần đo và đồng bộ GPU; xử lý ảnh bị loại khỏi phần đo thời gian.
8. **Chung kết:** cell kiểm tra `RUN_FINAL`. Mặc định giá trị là `False`; chỉ đổi thành `True` sau khi xem bảng validation và chốt cấu hình. Khi bật, notebook chạy T00/I00 và F01 trên ba seed. Nếu file test của seed đã tồn tại, notebook xác minh nó với CSV gốc rồi bỏ qua, không chạy inference test lần nữa.
9. **Workbook/eval:** chạy `eval.py score`, `eval.py grade` khi đủ dự đoán, rồi ghi `results.xlsx`.

Notebook dùng `tqdm` dạng văn bản cho danh sách thí nghiệm và epoch. Batch train chỉ in khoảng bốn dòng tiến độ mỗi epoch, không in thanh `train batches`/`val batches` liên tục; sau mỗi epoch có log loss/F1/thời gian và `checkpoint_last.pt` để resume ở lần chạy lại cùng runtime. Các biến `RUN_BACKBONES_B01_B02`, `RUN_BACKBONES_B03_B05`, `RUN_ABLATIONS`, `RUN_INFERENCE`, `RUN_FINAL` ở đầu notebook cho phép chạy theo giai đoạn. Nếu chỉ chạy nhóm B03–B05, đặt `RUN_BACKBONES_B01_B02 = False`, giữ `RUN_BACKBONES_B03_B05 = True` và chỉ chạy các cell setup, helper backbone, nhóm 2 rồi bảng backbone. Trước khi chạy ablation, cần khôi phục đủ kết quả của cả năm backbone. Không bật final trước khi các kết quả validation đã được xem xét.

Baseline mặc định dùng batch size 32 để giảm nguy cơ hết GPU memory trên các backbone lớn. Nếu OOM, giảm `BATCH_SIZE` trong cấu hình trước khi bắt đầu thí nghiệm; giữ cùng batch cho các backbone và ablation để so sánh công bằng, đồng thời ghi lại thay đổi trong output. Cấu hình chính dùng 12 epoch, AMP và ImageNet pretrained weights.

### Tiếp tục B03 trong notebook Kaggle hiện tại và giữ output cũ

Nếu editor Kaggle vẫn hiển thị output B01–B02, **không import đè** file `.ipynb` local và **không chọn Save & Run All**. Trước khi sửa, chọn **Save Version → Quick Save** để lưu bản notebook đang hiển thị; nếu cần giữ cả file trong `/kaggle/working`, kiểm tra tùy chọn lưu output files trong Advanced Settings. Quick Save không chạy lại notebook từ đầu ([Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)).

Sau khi kernel restart, chạy lại các cell cài đặt/cấu hình, ghi source, kiểm tra checksum, nhận diện Dataset, rồi cell định nghĩa `BASE_RECIPE`, `make_config`, `run_backbone_group`. Bỏ qua cell B01–B02 và các cell EDA/smoke test đã có output. Trong cell cấu hình, đặt `RUN_BACKBONES_B01_B02 = False`, `RUN_ABLATIONS = False`, `RUN_INFERENCE = False`, `RUN_FINAL = False`. Chèn **một code cell mới** ngay sau cell định nghĩa `run_backbone_group` và chạy:

```python
# Dùng một T4 chỉ trong lần chạy B03; không sửa output của B01/B02.
USE_DATA_PARALLEL = False
try:
    run_backbone_group([("B03", "convnext_tiny")])
finally:
    USE_DATA_PARALLEL = True
```

Cell này dùng được với notebook Kaggle phiên bản đã có `run_backbone_group` nhưng chưa có `B03_USE_DATA_PARALLEL`: `make_config` đọc `USE_DATA_PARALLEL` khi B03 bắt đầu. Nếu B03 hoàn thành và lưu đủ file, chạy tiếp cell nhóm B03–B05 cũ; nó sẽ in `Reuse B03` rồi chạy B04–B05 trên hai GPU. Không chạy lại cell B01–B02. Nếu mở **session mới** và `/kaggle/working/day2_output` không còn, cần gắn output cũ dưới `/kaggle/input` như mục 5 để kết quả B01–B02 xuất hiện trong bảng cuối. Output hiển thị trong cell và các file checkpoint là hai loại dữ liệu khác nhau.

## 5. Tìm và tải kết quả

Notebook ghi sản phẩm vào:

```text
/kaggle/working/day2_output/
├── results.xlsx
├── runs/                 # config, history, checkpoint_best.pt, checkpoint_last.pt
├── predictions/          # validation, test và uncalibrated test CSV
├── curves/               # đồ thị train/validation cho từng experiment và seed
├── eval/                 # kết quả score/grade từ eval.py
├── experiment_records.json
├── inference_records.json
├── latency_records.json
└── runtime.json
```

Trong phiên notebook, mở panel **Output** hoặc tab **Files** để tải `results.xlsx`, các CSV dự đoán, đường cong và log. Để giữ output sau khi kết thúc phiên, dùng **Save Version** với lựa chọn lưu output (hoặc tải các file về máy). **Save & Run All** chạy notebook trong một session mới; nếu chỉ muốn tiếp tục từ kernel đang mở, chạy riêng cell cần thiết thay vì khởi động lại toàn bộ thí nghiệm.

Các checkpoint được lưu trong `/kaggle/working/day2_output/runs/`. Nếu cần tiếp tục ở một session Kaggle mới, đặt `PACKAGE_RESUME_OUTPUT = True` rồi chạy lại cell cuối để tạo `/kaggle/working/day2_output.zip`. Tải archive này về máy, đưa nó vào một Kaggle Dataset rồi gắn Dataset đó làm input của session mới. Nếu Kaggle giữ nguyên file zip, điền đường dẫn vào `RESUME_ARCHIVE`; nếu Dataset upload đã giải nén nó thành thư mục `day2_output/`, điền đường dẫn thư mục đó vào `RESUME_OUTPUT_DIR`. Làm việc này trước khi chạy cell cấu hình runtime. Sau khi khôi phục, chạy các cell setup/data rồi chạy lại experiment cell phù hợp; checkpoint theo epoch có cùng cấu hình train sẽ được tiếp tục. Kaggle không giữ `/kaggle/working` như ổ đĩa local lâu dài sau khi session kết thúc.

Khi nạp lại output của lần chạy B01–B02 cũ, cell nhóm 1 nhận kết quả đã hoàn thành dù số DataLoader worker đổi từ 2 sang 0. Checkpoint huấn luyện dở của B03 với cấu hình worker hoặc DataParallel cũ không tương thích để resume đúng cùng chuỗi ngẫu nhiên; notebook sẽ bắt đầu lại B03 nếu không có record hoàn chỉnh.

## 6. Gửi kết quả để hoàn thiện nhận xét và report

Sau khi chạy xong, tải về và gửi:

- notebook `.ipynb` đã chạy có output các cell;
- `results.xlsx`;
- `predictions/` và thư mục `eval/`;
- `curves/`;
- nếu có thể, `experiment_records.json`, `inference_records.json`, `latency_records.json` và `runtime.json`.

Nhận xét và report sẽ được viết dựa trên những kết quả đã chạy và kiểm tra được. README, rubric và guide hiện tại không được xem là bằng chứng cho một metric hoặc kết quả Kaggle chưa chạy.

## 7. Lỗi thường gặp

- **Chỉ thấy một GPU hoặc DataParallel chưa active:** vào Settings → Accelerator, chọn **GPU T4 ×2**, khởi động lại session rồi chạy lại cell cấu hình. Chỉ bắt đầu train khi log xác nhận `GPU count=2` và `active for training=True`.
- **Không tìm được fold 0:** kiểm tra Dataset có `labels.csv` và đủ ba file `*_subset0.csv`; giữ chúng cùng một thư mục. Nếu có nhiều bản dữ liệu, điền `DATASET_SLUG`.
- **Thiếu ảnh:** upload `images.zip` hoặc toàn bộ JPG; notebook báo danh sách tên thiếu theo CSV. Không đổi `Filename` trong CSV để lách lỗi.
- **Không tải được pretrained weights hoặc `timm`:** bật Internet rồi khởi động lại cell cài package/model. Không sửa code để âm thầm chuyển sang khởi tạo scratch.
- **Hết bộ nhớ GPU:** giảm `BATCH_SIZE` và chạy lại các experiment với cấu hình nhất quán. Cấu hình thay đổi sẽ được lưu riêng; không trộn số liệu giữa hai batch size mà không ghi rõ.
- **Kaggle báo `tried to allocate more memory than is available`:** đây là lỗi RAM hệ thống. Notebook hiện dùng `NUM_WORKERS = 0`, một PyTorch CPU thread và progress dạng văn bản để giảm tải. Log B03 in thêm `process_rss` và `cgroup` tại bốn mốc mỗi epoch; gửi các dòng này và trạng thái checkpoint nếu RAM vẫn tăng đến giới hạn. CPU hiển thị 200% không tự nó chứng minh nguyên nhân sập kernel.
- **`Error displaying widget: model not found`:** upload notebook local mới đã tạo lại; nó dùng `tqdm` dạng văn bản nên không cần widget progress của Kaggle.
- **Notebook bị ngắt:** checkpoint theo epoch nằm trong `runs/<exp_id>/seed<k>/`. Giữ output trước khi session kết thúc; sau khi đưa checkpoint vào session mới, chạy lại cell cấu hình/đồng bộ dữ liệu và experiment cell tương ứng để resume.
- **`Missing resume output directory` hoặc lỗi `RESUME_*`:** lần chạy mới hãy để trống cả `RESUME_OUTPUT_DIR` và `RESUME_ARCHIVE`. Khi resume, gắn Dataset chứa output cũ rồi điền đúng một biến bằng đường dẫn tương ứng dưới `/kaggle/input/`.
- **Test prediction đã tồn tại:** notebook cố ý không ghi đè. Dùng đúng file đã sinh và không đổi recipe rồi chạy lại test; nếu đã phát hiện lỗi sau test, ghi rõ ảnh hưởng trong report theo hướng dẫn lab.
