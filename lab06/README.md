# LAB 06 - Attention Mechanism


## Thứ tự cell notebook

Notebook gồm **19 code cells**, thực thi từ 1 tới 19 trong một kernel mới:

| Cell | Mục đề | Nội dung |
|---|---|---|
| 1 | 12 | Cấu hình và xác nhận prediction |
| 2 | 13 | Scaled dot-product đủ các bước |
| 3 | 15 | Temperature, max weight, entropy, nhận xét số liệu tự động |
| 4 | 16 | Ba output scores / attention_weights / context |
| 5 | 17 | Stable softmax |
| 6 | 18 | Token / Score / Attention và bar plot |
| 7 | 19 | Thay query và đo chênh lệch weights |
| 8 | 20 | Tạo Q/K/V từ X, hiển thị từng ma trận và context |
| 9 | 22 | Attention matrix và heatmap |
| 10 | 23 | Hai câu cat/book, rows và top keys cho token yêu cầu; untrained |
| 11 | 24a | Dữ liệu, splits, train-only vocabulary |
| 12 | 24b | Train, histories và validation |
| 13 | 24c | Bảng Model / Accuracy / F1 / Training time / Parameters; confusion matrices |
| 14 | 25 | Length 10/100/500/1000, số phần tử, storage và benchmark |
| 15 | 26 | Ba câu application, sentiment/probability/heatmaps; bổ sung trained cat/book rows |
| 16 | 26, 27 | 3 đúng/3 sai: full text, nhãn, weights, perturbation; chưa gán nguyên nhân |
| 17 | 28 | Đo hành vi padding và đảo thứ tự |
| 18 | 29 | T=1000 / 10000, số phần tử và tỷ lệ |
| 19 | 32 | Kiểm chứng NumPy/PyTorch, metadata, trạng thái deliverables |

Các mục chỉ tính tay hoặc tự trả lời không có cell giả để giải hộ. Mục 21 hoàn toàn thuộc `calculations.md`. Cell 8 dùng một sequence khác và ma trận demo có seed để minh họa API/heatmap; không phải lời giải mục 21.

## Điều kiện thí nghiệm và giới hạn để người học tự xét

- IMDb Movie Reviews, negative=0, positive=1. Dùng lại cleaning và manifest LAB05/LAB04: 19.920 train, 4.981 validation, 24.675 test; kiểm nhãn, hash token document và không overlap giữa tập. Hai parquet được kiểm SHA256, revision cố định trong `preprocessing.py`. Không dùng nhãn test để chọn model/hyperparameters.
- Vocabulary fit train only, tối đa 20.000 gồm PAD=0, UNK=1, min count=2; tokenizer giữ negation; right truncation giữ prefix 100 tokens; padding bị mask. Có bảng tỷ lệ token được giữ/truncated để người học phân tích giới hạn input.
- Cả hai model được train lại trong phiên LAB06: seed=42; embedding=64, hidden/key/value=64, batch=128, Adam lr=0.001, 6 epochs, clipping=1.0. Embedding khởi tạo giống nhau và batch order theo cùng seed. Không dùng timing lịch sử LAB05 làm timing mới.
- Baseline: Embedding -> 1-layer unidirectional LSTM -> final valid hidden state -> Linear classifier. Attention: Embedding -> WQ/WK/WV -> full single-head self-attention -> masked mean pooling -> Linear classifier. Không sử dụng built-in attention API trong core, không Transformer/multi-head. Không thêm positional encoding vì bài dành phần này cho LAB07; người học cần tự xét hệ quả với thứ tự token dựa trên probe cell 17. Không tuyên bố pipeline hiện tại tự có mọi property trong bảng lý thuyết mục 28.
- Dùng min validation BCE để chọn checkpoint trong 6 epochs, test chỉ evaluation sau khi chọn. Cấu hình cố định trước train; chạy một seed không phải kết luận thống kê nhiều lần.
- Training time là tổng thời gian training batches, có đồng bộ CUDA, không gồm validation/evaluation; `fit_wall_seconds` và validation time lưu riêng. Accuracy, precision, recall, F1 positive=1, threshold=0.5; có macro F1. Parameters bao gồm embedding và classifier.
- NumPy và PyTorch attention đều thực hiện phép toán trực tiếp; cell 19 đối chiếu bằng chính Q/K/V đã học. Kiểm padding invariance. Mục 17 kiểm input lớn và sự hữu hạn. Mục 15 hiện có nhận xét số liệu sinh tự động bằng code theo yêu cầu bổ sung của người học; không tự viết lý giải lý thuyết, giải thích heatmap hay hypothesis lỗi.
- Mục 18-19 dùng seeded random keys/values/queries; query A/B không được gọi là query sentiment/topic đã học. Mục 22-23 trước train dùng embeddings/projections ngẫu nhiên; tên hình ghi UNTRAINED. Mục 26 bổ sung hai câu cat/book với weights từ classifier sentiment đã học; đây không phải parser được train cho ngữ pháp.
- Full attention matrix lưu hàng=query, cột=key. `mean_query_attention` là trung bình theo các hàng query hợp lệ, phù hợp phép mean pooling hiện dùng; biểu đồ tổng hợp này không thay thế ma trận đầy đủ. CSV/NPY giữ toàn bộ values và text, notebook có thể cắt bớt text khi hiển thị.
- Sáu mẫu lấy cơ học 3 đúng đầu tiên + 3 sai đầu tiên để tiện chọn; không phải đã được phân loại nguyên nhân. Có thể chọn mẫu khác trong pool. Perturbation thay liên tiếp mỗi 10 token bằng UNK, giữ độ dài/vị trí; người học tự xét hạn chế và đặt hypothesis. Không tạo lời kết luận attention/causal attribution thay người học.
- Bảng chi phí chỉ tính bytes của **một** dense matrix float32, không phải peak memory toàn model, gradients, batch hay optimizer. Benchmark forward batch=1, d=64, 5 warmups + 30 repeats; số đo ngắn có overhead, không bắt buộc tăng đúng tỷ lệ lý thuyết. Không cấp phát ma trận 10000².


- `calculations.md` -> `scans/calculations_scan.pdf`: mục 6, 8, 9, 10, 11, 14, **21**.
- `prediction.md` -> `scans/prediction_scan.pdf`: bản 5 prediction mục 12 đã viết **trước code**, giữ nguyên không sửa theo kết quả.
- `error_analysis.md` -> `scans/error_analysis_scan.pdf`: mục 26 và 27, có ghi sample ID/nhãn/weights/evidence tương ứng.
- `reflection.md` -> `scans/reflection_scan.pdf`: các câu giải thích/nhận xét còn lại theo checklist trong file.



## Chạy lại và kiểm chứng

1. Cài dependencies: `python -m pip install -r requirements.txt`. Bản torch CUDA phải phù hợp máy nếu muốn dùng GPU; code có fallback CPU.
2. Tự hoàn thành prediction W6 trước chạy, cập nhật `prediction_confirmation.json` cho đúng thực tế người chạy. File xác nhận hiện tại ghi lời xác nhận của người học trong phiên 22/10/2026, không thay bản scan prediction.
3. Mở `self_attention.ipynb` với working directory là `lab06/`, Restart Kernel rồi Run All **từ trên xuống**. Không chạy cell giữa trước hoặc giữ output của kernel cũ.
4. Giữ `attention_numpy.py`, `experiment_utils.py`, `preprocessing.py`, `data/`, `outputs/` nếu cần tái lập và chạy application. Checkpoints nằm trong `outputs/checkpoints/`.

```powershell
python -X utf8 -c "import nbformat; from nbclient import NotebookClient; from pathlib import Path; p=Path('self_attention.ipynb'); n=nbformat.read(p,4); c=NotebookClient(n,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}); c.execute(); nbformat.write(n,p)"
```

Kết quả kiểm tra thực thi được lưu ở `outputs/notebook_verification.json` sau khi chạy; cấu hình/checksum ở `outputs/run_metadata.json`. `outputs/requirements_audit.csv` đối chiếu yêu cầu output của từng mục có code với vị trí cell và file bằng chứng. Kiểm tra kỹ thuật không thay người học kiểm chứng nội dung và tự giải thích code. Bài nộp chỉ hoàn chỉnh sau khi người học bổ sung đủ file tự viết/scan, làm bài tính tay 21 với dữ kiện hợp lệ, và tự hoàn thành các câu giải thích/phân tích.

## AI assistance statement - mục 30

- **Tool:** ChatGPT.
- **Purpose:** Hỗ trợ NumPy/PyTorch API, boilerplate training/evaluation, visualization/plotting, debugging, documentation và kiểm tra thực thi.
- **Generated:** Nhận xét số liệu tự động ở mục 15 và các thống kê quan sát khác theo yêu cầu bổ sung của người học (chỉ dùng giá trị thực chạy, không tự gán nguyên nhân lỗi hoặc trả lời lý thuyết). Code API trong `attention_numpy.py`, code notebook, phần self-attention PyTorch/helper thí nghiệm, khung checklist Markdown và liên kết PDF scan, requirements, script tạo/chạy notebook trong `tmp/`. CSV/hình/checkpoint là output từ code chạy thật. Không sinh lời giải bộ số tính tay mục 6/8-11/14/21, prediction, giải thích Q/K/V thay người học, interpretation heatmap, error analysis, reflection hoặc individual check.
- **Modified:** Tạo `lab06/`; tái sử dụng `preprocessing.py`, dữ liệu và split manifest từ LAB04/LAB05, chỉnh helper sequence thành LSTM/Attention phục vụ LAB06. 
