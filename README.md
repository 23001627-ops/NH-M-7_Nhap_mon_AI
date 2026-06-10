# NH-M-7_Nhap_mon_AI
ĐỀ TÀI: Xây dựng hệ thống hỗ trợ phân tích phản hồi sinh viên nhằm cải thiện chất lượng giảng dạy sử dụng kỹ thuật phân loại cảm xúc văn bản
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![Library](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UIT--VSFC-green.svg)](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)

Hệ thống ứng dụng Trí tuệ Nhân tạo và Xử lý ngôn ngữ tự nhiên (NLP) nhằm tự động nhận diện và phân loại cảm xúc từ các phản hồi văn bản tự do của sinh viên. Hệ thống phân tách dữ liệu thành 3 nhóm trạng thái: **Tích cực (Positive)**, **Tiêu cực (Negative)**, và **Trung lập (Neutral)**. Đặc biệt, ứng dụng web còn đi sâu vào phân tích nguyên nhân và tự động đưa ra giải pháp thực tế để cải thiện chất lượng giảng dạy.

---

## 📌 Pipeline Tổng Quan Hệ Thống

```mermaid
graph TD
    A[Phản hồi của sinh viên] --> B[Tiền xử lý văn bản]
    B --> C[Trích xuất đặc trưng TF-IDF]
    C --> D[Huấn luyện / Dự đoán bằng Mô hình Học máy]
    D --> E{Phân loại Cảm xúc}
    E -->|Tiêu cực| F[Phân tích Nhóm Nguyên nhân sâu]
    F --> G[Đề xuất Giải pháp Cải thiện]
    E -->|Tích cực / Trung lập| H[Trực quan hóa thống kê] xúc & Trực quan hóa]
    F --> G[Đề xuất cải thiện giảng dạy]
