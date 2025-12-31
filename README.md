# sleep-quality-prediction
# 😴 Dự đoán Chất lượng Giấc ngủ (Sleep Quality Prediction)

## 1. Giới thiệu đề tài
**Bài toán:** Chất lượng giấc ngủ đóng vai trò quan trọng đối với sức khỏe thể chất và tinh thần. Dự án này xây dựng một hệ thống Machine Learning để dự đoán chất lượng giấc ngủ của người dùng dựa trên các chỉ số sức khỏe và thói quen sinh hoạt.

**Mục tiêu:**
* Phân tích các yếu tố ảnh hưởng đến giấc ngủ (Stress, BMI, Hoạt động thể chất...).
* Xây dựng mô hình phân lớp để dự đoán chất lượng giấc ngủ theo 3 mức độ: **Kém, Trung bình, Tốt**.
* Xây dựng ứng dụng Web giúp người dùng tự kiểm tra và nhận lời khuyên cải thiện.

## 2. Dataset
* **Nguồn dữ liệu:** Tập dữ liệu `Sleep_health_and_lifestyle_dataset.csv` (đã bao gồm trong thư mục `data/`).
* **Kích thước:** ~374 bản ghi.
* **Mô tả các đặc trưng (Features):**
    * `Gender`: Giới tính.
    * `Age`: Tuổi.
    * `Occupation`: Nghề nghiệp.
    * `Sleep Duration`: Thời gian ngủ (giờ/ngày).
    * `Quality of Sleep`: Điểm chất lượng giấc ngủ (Target gốc: thang 1-10).
    * `Physical Activity Level`: Mức độ hoạt động thể chất (phút/ngày).
    * `Stress Level`: Mức độ căng thẳng (thang 1-10).
    * `BMI Category`: Chỉ số khối cơ thể (Normal, Overweight, Obese).
    * `Blood Pressure`: Huyết áp (dạng chuỗi "126/83").
    * `Heart Rate`: Nhịp tim (bpm).
    * `Daily Steps`: Số bước chân hàng ngày.
    * `Sleep Disorder`: Rối loạn giấc ngủ (None, Insomnia, Sleep Apnea).

## 3. Pipeline (Quy trình thực hiện)
Quy trình xử lý từ dữ liệu thô đến ứng dụng thực tế:

1.  **Tiền xử lý (Preprocessing):**
    * Xử lý giá trị thiếu (Missing Values) ở cột `Sleep Disorder` (điền "None").
    * Feature Engineering: Tách cột `Blood Pressure` thành 2 cột số `Systolic` (Tâm thu) và `Diastolic` (Tâm trương).
    * Label Encoding: Mã hóa dữ liệu phân loại (`Gender`, `Occupation`, `BMI`, `Sleep Disorder`).
    * Target Binning: Gom nhóm `Quality of Sleep` thành 3 nhãn: **0 (Kém)**, **1 (Trung bình)**, **2 (Tốt)**.
2.  **Huấn luyện (Training):** Chia tập dữ liệu Train/Test (tỷ lệ 80/20) và huấn luyện mô hình.
3.  **Đánh giá (Evaluation):** Kiểm tra độ chính xác (Accuracy), Confusion Matrix và vẽ Learning Curve.
4.  **Triển khai (Inference):** Tích hợp mô hình vào ứng dụng web bằng Streamlit.

## 4. Mô hình sử dụng
Nhóm nghiên cứu và áp dụng hai thuật toán:

* **Decision Tree (Cây quyết định):**
    * *Lý do chọn:* Mô hình đơn giản, dễ giải thích, giúp trực quan hóa quy trình ra quyết định.
    * *Cấu hình:* `criterion='entropy'`, `max_depth=3`.
* **Random Forest (Rừng ngẫu nhiên):**
    * *Lý do chọn:* Khắc phục nhược điểm Overfitting của Decision Tree, cho độ chính xác cao hơn và ổn định hơn trên tập dữ liệu nhỏ.
    * *Kết quả:* Được chọn làm mô hình chính cho ứng dụng Demo.

## 5. Kết quả thực nghiệm
Dựa trên việc huấn luyện và kiểm thử mô hình trên tập dữ liệu đã chia (80% Training - 20% Testing), nhóm thu được kết quả chi tiết như sau:

### Bảng so sánh độ chính xác (Model Performance)

| Mô hình | Độ chính xác trên tập Train | Độ chính xác trên tập Test (Accuracy) | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Decision Tree** | ~ 91.5% | **89.0%** | Mô hình hoạt động khá tốt nhưng có dấu hiệu dao động nhẹ, độ phức tạp thấp. |
| **Random Forest** | ~ 95.2% | **93.5%** | **Mô hình tốt nhất**. Khả năng tổng quát hóa cao, ít bị Overfitting hơn so với cây quyết định đơn lẻ. |

> *Lưu ý: Kết quả có thể thay đổi nhẹ tùy thuộc vào `random_state` khi chia dữ liệu.*

### Chi tiết đánh giá (Classification Report)
Mô hình **Random Forest** cho kết quả phân lớp rất tốt trên cả 3 nhãn dự đoán:
* **Nhóm 0 (Chất lượng Kém):** Recall đạt mức cao, giúp phát hiện đúng hầu hết những người có vấn đề về giấc ngủ (tránh bỏ sót bệnh).
* **Nhóm 1 (Trung bình) & Nhóm 2 (Tốt):** Độ chính xác (Precision) cao, ít bị nhầm lẫn giữa hai nhóm này.

### Phân tích biểu đồ (Visual Analysis)
* **Learning Curve (Đường cong học tập):** * Khoảng cách giữa đường *Training Score* và *Validation Score* của Random Forest hẹp dần khi số lượng mẫu tăng lên.
    * Điều này chứng tỏ mô hình không bị **Overfitting** (học vẹt) hay **Underfitting** (học chưa tới), đảm bảo độ tin cậy khi áp dụng vào thực tế.
* **Feature Importance (Mức độ quan trọng của đặc trưng):**
    * Các yếu tố ảnh hưởng lớn nhất đến kết quả dự đoán lần lượt là: **Sleep Duration** (Thời lượng ngủ), **Stress Level** (Mức độ căng thẳng) và **BMI** (Chỉ số cơ thể).

### Kết luận
Dựa trên các chỉ số trên, nhóm quyết định lựa chọn **Random Forest** làm mô hình chính thức (Back-end) cho ứng dụng dự đoán.

## 6. Hướng dẫn cài đặt và chạy dự án

### Bước 1: Chuẩn bị môi trường
Yêu cầu máy tính đã cài đặt Python 3.8+.
1.  **Tạo và kích hoạt môi trường ảo (Khuyên dùng):**
    * Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

### Bước 2: Chạy Training (Huấn luyện mô hình)
Phần này giúp bạn xem lại quy trình phân tích dữ liệu (EDA), tiền xử lý và huấn luyện lại mô hình nếu cần.

1.  **Khởi động Jupyter Notebook:**
    Tại terminal (đang ở thư mục gốc), chạy lệnh:
    ```bash
    jupyter notebook
    ```
2.  **Mở file:**
    Trình duyệt sẽ mở ra. Truy cập vào thư mục `demo/` và chọn file `Sleep_health.ipynb`.
3.  **Thực thi:**
    Chọn menu **Cell > Run All** để chạy toàn bộ các bước từ đọc dữ liệu đến vẽ biểu đồ đánh giá.

### Bước 3: Chạy Demo (Ứng dụng dự đoán)
Đây là ứng dụng giao diện Web cho phép người dùng nhập thông tin và nhận kết quả dự đoán chất lượng giấc ngủ.

1.  **Chạy lệnh Streamlit:**
    Tại terminal (đang ở thư mục gốc), chạy lệnh:
    ```bash
    streamlit run app/app.py
    ```
2.  **Sử dụng:**
    * Trình duyệt sẽ tự động mở địa chỉ (thường là `http://localhost:8501`).
    * Nhập các chỉ số sức khỏe vào thanh bên trái (Sidebar).
    * Nhấn nút dự đoán để xem kết quả và lời khuyên.

## 7. Cấu trúc thư mục dự án
Dự án được tổ chức theo cấu trúc chuẩn như sau:

```text
Sleep_Quality_Project/
│
├── app/
│   └── app.py                      # Source code chính của ứng dụng Web (Streamlit)
│
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv  # Dữ liệu gốc sử dụng cho huấn luyện
│
├── demo/
│   └── Sleep_health.ipynb          # Notebook dùng để phân tích dữ liệu (EDA) và thử nghiệm mô hình
│
├── reports/
│   └── slepp-quality-prec_Report.docx               # Báo cáo
│
├── slides/
│   └── Sleep_quality.pptx   # Slide báo cáo
│
├── venv/                           # Thư mục môi trường ảo
├── .gitignore                      # Cấu hình các file GitHub cần bỏ qua
├── requirements.txt                # Danh sách các thư viện Python cần thiết
└── README.md                       # Tài liệu hướng dẫn sử dụng
```

## 8. Tác giả
* Đỗ Văn Ngọc: Mã sinh viên 12423026 lớp 124231
* Đoàn Nhật Anh: Mã sinh viên 12423043 lớp 124231
