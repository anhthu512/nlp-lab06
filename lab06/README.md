# LAB 06 - Attention Mechanism

Đề: W6.pdf, 20 trang, mục 1-35. Hạn nộp ghi trong đề: **23:59 ngày 28/10/2026**.

`self_attention.ipynb` chỉ có code cells và output thực thi. Những câu yêu cầu giải thích, nhận xét, tính tay, prediction và phân tích lỗi được để cho người học tự làm theo mục 30. Checklist dưới đây chỉ liệt kê việc cần làm, không phải câu trả lời để nộp.

## Phân công đầy đủ theo mục đề

| Mục W6 | Code / output | Người học tự viết, không dùng AI trả lời | Cần dùng output để tự nhận xét / phân tích? | File nhận scan |
|---|---|---|---|---|
| 1 | Bối cảnh, không có code riêng | Không có bài nộp riêng | Không | - |
| 2 | Mục tiêu LO1-LO12, không có cell riêng | Tự kiểm tra mức hiểu các mục tiêu | Không bắt buộc | - |
| 3 | Lịch hoạt động, không có code riêng | Không có câu trả lời riêng | Không | - |
| 4 | Không có code riêng | Câu hỏi về representation từ hidden state cuối | Không | `reflection.md` |
| 5 | Không code giải ví dụ tính toán | Giải thích attention score và Q/K/V bằng lời mình | Không | `reflection.md` |
| 6 | Không code giải bộ số ví dụ | Tự tính softmax [2,1,0], xác định token nhận weight lớn nhất | **Không**, tính từ đề | `calculations.md` |
| 7 | Không có cell giải bài tính tay | Tự hiểu / giải thích context vector, liên hệ Q/K/V | Không | `reflection.md` |
| 8 - Bài 1 | Không code giải bộ số đề | Tính từng dot product và toàn bộ score vector | **Không** | `calculations.md` |
| 9 - Bài 2 | Không code giải bộ số đề | Tính softmax, kiểm tổng bằng 1, giải thích score lớn / weight lớn | **Không** | `calculations.md` |
| 10 - Bài 3 | Không code giải bộ số đề | Tính context và giải thích đóng góp của các value | **Không** | `calculations.md` |
| 11 - Bài 4 | Không code giải bộ số đề | Toàn bộ q -> scores -> softmax -> weights -> context; trình bày từng bước, không thư viện attention | **Không** | `calculations.md` |
| 12 | Cell 1 kiểm tra xác nhận đã viết prediction trước code | Tự viết đủ 5 Prediction / Reason / Confidence; người học đã xác nhận hoàn thành trước lần chạy này | **Không; phải viết trước experiment**, giữ nguyên bản ban đầu | `prediction.md` |
| 13 | Cell 2: hiển thị Q/K/V, QKᵀ, scaled scores, attention weights, context; thêm thống kê theo dimension | Tự giải thích scaled dot-product và vai trò scaling | Không bắt buộc cho lý thuyết; **có** nếu nhận xét kết quả thực nghiệm | `reflection.md` |
| 14 | Không code giải bộ số scaling trong đề | Tự tính scaled / unscaled scores, softmax, so sánh độ tập trung | **Không**, dùng dữ kiện đề | `calculations.md` |
| 15 | Cell 3: T=0.5/1/2, cùng scores, max weight, entropy, bảng đúng cột đề (Nhận xét tự động bằng số liệu), bảng weights và đồ thị | Notebook có nhận xét số liệu tự động theo yêu cầu mới của người học; phần lý giải và đối chiếu prediction tự viết | **Có**: `temperature.csv`, `temperature.png` | `reflection.md` |
| 16 | `attention_numpy.py`; cell 4 hiển thị đầy đủ scores, attention_weights, context và Q/K/V; lưu NPZ/CSV; kiểm tra shape, tổng hàng, đối chiếu phép toán tensor | Sinh viên phải tự đọc, kiểm tra, hiểu và giải thích từng bước; tự chỉnh bằng hiểu biết của mình nếu cần | **Có** khi nêu kết quả kiểm chứng; không cần output cho giải thích cơ chế | `reflection.md`, AI statement trong README |
| 17 | Cell 5: stable softmax, test [1000,1001,1002], kiểm hữu hạn và shift invariance | Tự giải thích vì sao trừ max không đổi softmax | **Không bắt buộc** cho lập luận; **có** nếu dẫn kiểm thử runtime | `reflection.md` |
| 18 | Cell 6: đúng sequence đề; query giả lập; token / raw score / scaled score / weight, bar plot | Tự quan sát phân phối trọng số | **Có**, bảng và hình mục 18; đây là dữ liệu giả lập chưa học | `reflection.md` |
| 19 | Cell 7: giữ nguyên sequence, K, V, thay query A/B; bảng và biểu đồ | Tự so sánh attention khi query đổi; không gán ý nghĩa sentiment/topic cho random query | **Có**, `query_comparison.csv/png` | `reflection.md` |
| 20 | `self_attention()` NumPy và classifier dùng Q/K/V từ cùng sequence | Tự giải thích self-attention và phân biệt với attention tổng quát | Không bắt buộc; shape output chỉ là hỗ trợ | `reflection.md` |
| 21 | **Không code giải bài này**, theo yêu cầu người học | Bài tính tay XWQ, XWK, XWV, QKᵀ, scaled softmax, AV; PDF không cho số cụ thể X/WQ/WK/WV, cần dùng dữ kiện giảng viên cung cấp riêng nếu có | **Không**, không dùng output demo thay lời giải | `calculations.md` |
| 22 | Cell 9: self-attention heatmap trên dữ liệu demo độc lập, không dùng bộ số tính tay mục 21 | Tự mô tả ý nghĩa hàng/cột và phân phối attention; không coi bảng ví dụ trong đề là kết quả model | **Có** khi nhận xét heatmap | `reflection.md` |
| 23 | Cell 10: hai câu cat/book theo đề, heatmaps, toàn bộ hàng cat/adopted/sleeping; cell 15 bổ sung heatmaps classifier đã học | Tự mô tả 3 token được hỏi; phân biệt pattern quan sát với giải thích ngôn ngữ | **Có**, bắt buộc xem đúng nguồn weights (untrained hoặc trained) | `reflection.md` |
| 24 | Cells 11-13: IMDb, train LSTM và Attention, bảng accuracy/F1/training time/parameters; curves, confusion matrices | Tự so sánh representation; đối chiếu prediction ban đầu, xem giới hạn cấu hình | **Có**, `results.csv`, validation và histories | `reflection.md` |
| 25 | Cell 14: bảng số phần tử cho 10/100/500/1000, thêm 10000 ở mục 29, storage và benchmark | Tự giải thích computational cost khi sequence dài | **Có** cho nhận xét số đo/bảng; lý thuyết tự lập luận | `reflection.md` |
| 26 | Application raw text -> tokenizer -> embedding -> self-attention -> pooled context -> classifier; đúng 3 câu đề, prediction/probability/heatmaps; thêm pool test có nhãn | Tự chọn ít nhất 1 prediction đúng + 1 sai, phân tích; tự xác định nhãn kỳ vọng cho ba câu tự nhập | **Có**, text model thực nhận, xác suất, heatmap; không mặc định mọi câu đều đúng/sai theo cảm tính | `error_analysis.md` |
| 27 | Toàn bộ dự đoán test, 3 đúng + 3 sai chưa ghi nguyên nhân, full attention matrices, token tables, span perturbations | Tự chọn 3 đúng + 3 sai, phân loại lỗi từ danh sách đề, đặt hypothesis và đối chiếu hành vi; không để AI gán nguyên nhân | **Có**, bắt buộc text/nhãn/prediction/weights/truncation; perturbation chỉ là bằng chứng phụ | `error_analysis.md` |
| 28 | Cell 17 có padding/order behavior checks; bảng LSTM/Attention mục 24 làm evidence | Tự trả lời attention giải quyết gì của recurrent processing và tạo vấn đề gì mới; xét cấu hình attention hiện dùng | **Có** cho thực nghiệm; cơ chế cần tự lập luận | `reflection.md` |
| 29 | Bảng chi phí gồm T=1000 và 10000; chỉ tính kích thước, không cấp phát ma trận 10000² | Tự nhận xét chi phí và trade-off; không kết luận Attention luôn thay RNN | **Có** | `reflection.md` |
| 30 | AI assistance statement ở cuối README | Người học bổ sung Modified/Verified của chính mình và tự giải thích Q/K/V, QKᵀ, softmax, AV | Không bắt buộc; dùng kiểm chứng code nếu đề cập | `README.md` |
| 31 | Không code trả lời individual check | Tự ôn toàn bộ câu hỏi mục 31; AI không trả lời hộ | Không bắt buộc; có thể xem output để ôn các câu thực nghiệm | Không yêu cầu file nộp riêng |
| 32 | Giữ đúng 10 deliverables; có thêm helper/data/output phục vụ tái lập | Tự bổ sung scan để bài nộp hoàn chỉnh | Có khi kiểm đủ artifact | 4 `.md` và 4 PDF scan |
| 33 | Rubric 100 điểm, không experiment riêng | Tự kiểm tra đủ các thành phần và phần tự viết | Có đối với mục thực nghiệm | - |
| 34 | Checklist peer assessment, không sinh review giả | Tự/nhờ reviewer kiểm tra theory, calculation, implementation, experiment, analysis, understanding | Có khi kiểm experiment / analysis | Có thể ghi vào `reflection.md` nếu cần |
| 35 | Không thêm Transformer / multi-head experiment | Tự suy ngẫm mạch nối LAB05 -> LAB06 -> LAB07 nếu viết phần kết; không có câu trả lời AI | Không bắt buộc | `reflection.md` |

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

## Nguồn output cho phần tự viết

| Phần cần tự viết | Evidence |
|---|---|
| 13 | Output cell 2; `outputs/section13_scaling_demo.npz`, `scaling.csv` |
| 15 | Output cell 3; `outputs/temperature.csv`, `temperature_weights.csv`, `temperature.png`; Nhận xét số liệu được code tạo tự động; phần lý giải người học tự viết |
| 16 | Output cell 4; `outputs/core_attention_outputs.npz`, `core_scores.csv`, `core_attention_weights.csv`, `core_context.csv`, `core_implementation_verification.json` |
| 17 | Output cell 5; `outputs/numerical_stability.csv`; `verification.json` |
| 18-19 | `simulated_query_A.csv`, `simulated_distribution.png`, `query_comparison.csv/png`, `simulated_qkv.npz` |
| 22 | `numpy_self_attention_demo.npz`, `numpy_attention_matrix.csv`, `numpy_self_attention_heatmap.png` |
| 23 | `section23_untrained_1/2.png`, `section23_matrix_1/2.csv`, `section23_token_rows.csv`, `section23_top_keys.csv`; trained bổ sung: `section23_trained_token_rows.csv`, `section23_trained_top_keys.csv`, `section23_trained_1/2_heatmap.png`, `_matrix.csv`, `_tokens.csv` |
| 24 | `results.csv`, `confusion_matrix.png`, `outputs/validation_results.csv`, `learning_history.csv`, `learning_curves.png`, `retention.csv` |
| 25, 29 | `sequence_length_cost.csv`, `sequence_length_benchmark.csv`, `sequence_length_ratios.csv`, `section29_attention_cost.csv`, `sequence_length_cost.png` |
| 26 | `application_predictions.csv`, `application_1/2/3_tokens.csv`, `application_1/2/3_matrix.csv`, `attention_heatmap.png`, `application_2/3_heatmap.png`; `application_labeled_candidates.csv` |
| 27 | `predictions_attention.csv`, `error_analysis_candidates.csv`, `six_examples_unannotated.csv`, `error_analysis_template.csv`, `error_attention_artifacts.csv`, `error_candidate_1..6_tokens.csv`, `_matrix.csv`, `_attention.npy`, `error_candidates_attention.png`, `span_perturbations.csv` |
| 28 | `results.csv`, `model_behavior_checks.csv`, `verification.json` |

Các tên không ghi thư mục ở bảng trên đều nằm trong `outputs/`, trừ ba file gốc `results.csv`, `attention_heatmap.png`, `confusion_matrix.png`.

## Scan PDF và chèn vào Markdown

Người học tự viết, scan thành PDF thật, rồi đặt vào `scans/`:

- `calculations.md` -> `scans/calculations_scan.pdf`: mục 6, 8, 9, 10, 11, 14, **21**.
- `prediction.md` -> `scans/prediction_scan.pdf`: bản 5 prediction mục 12 đã viết **trước code**, giữ nguyên không sửa theo kết quả.
- `error_analysis.md` -> `scans/error_analysis_scan.pdf`: mục 26 và 27, có ghi sample ID/nhãn/weights/evidence tương ứng.
- `reflection.md` -> `scans/reflection_scan.pdf`: các câu giải thích/nhận xét còn lại theo checklist trong file.

Mỗi `.md` cần có link tương đối và thẻ `<object>` chèn PDF. Một số viewer/GitHub không render PDF trong Markdown; link vẫn mở được khi đã có PDF ở đúng vị trí. Phải nộp cả file `.md` **và** PDF scan; link không chứa nội dung scan. Có thể scan nhiều trang trong mỗi PDF. Không tạo scan giả hoặc điền câu trả lời thay người học. Scan chèn `.md` là quy cách người học yêu cầu thêm; mục 32 của đề liệt kê các `.md`, không tự yêu cầu định dạng scan.

Lần kiểm tra hiện tại thấy bốn file `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` không còn trong `lab06/`; đợt sửa notebook này giữ nguyên trạng thái các file đó. Bảng trạng thái cell 19 và `outputs/artifact_audit.json` ghi rõ file còn thiếu, không coi đủ output code là đã hoàn chỉnh phần bài tự viết.

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

- **Tool:** ChatGPT / Codex.
- **Purpose:** Hỗ trợ NumPy/PyTorch API, boilerplate training/evaluation, visualization/plotting, debugging, documentation và kiểm tra thực thi.
- **Generated:** Nhận xét số liệu tự động ở mục 15 và các thống kê quan sát khác theo yêu cầu bổ sung của người học (chỉ dùng giá trị thực chạy, không tự gán nguyên nhân lỗi hoặc trả lời lý thuyết). Code API trong `attention_numpy.py`, code notebook, phần self-attention PyTorch/helper thí nghiệm, khung checklist Markdown và liên kết PDF scan, requirements, script tạo/chạy notebook trong `tmp/`. CSV/hình/checkpoint là output từ code chạy thật. Không sinh lời giải bộ số tính tay mục 6/8-11/14/21, prediction, giải thích Q/K/V thay người học, interpretation heatmap, error analysis, reflection hoặc individual check.
- **Modified:** Tạo `lab06/`; tái sử dụng `preprocessing.py`, dữ liệu và split manifest từ LAB04/LAB05, chỉnh helper sequence thành LSTM/Attention phục vụ LAB06. Không sửa bài lab trước. Người học tự ghi thêm những chỉnh sửa của mình sau phiên này.
- **Verified:** Kiểm chứng máy ghi trong notebook/output JSON. Người học **chưa được xem là đã tự kiểm chứng chỉ vì code chạy được**; cần tự đọc, kiểm tra, hiểu và giải thích `scaled_dot_product_attention`, Q/K/V, QKᵀ, stable softmax, AV, masking, pooling, training/evaluation; bổ sung xác nhận/chỉnh sửa của bản thân. Mục 16 yêu cầu sinh viên tự implement bằng NumPy: code ở đây có AI hỗ trợ API, phải khai báo trung thực và tự rà soát, không trình bày như code hoàn toàn tự viết không có hỗ trợ.
