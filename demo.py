import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Phân Tích Phản Hồi Sinh Viên",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Hệ Thống Phân Tích Cảm Xúc Phản Hồi Sinh Viên")
st.markdown("### Hỗ Trợ Nâng Cao Chất Lượng Giảng Dạy Bằng Trí Tuệ Nhân Tạo")
st.write("---")

# ==========================================
# 2. ĐỊNH NGHĨA HÀM TIỀN XỬ LÝ & TỪ ĐIỂN
# ==========================================
VIETNAMESE_STOPWORDS = set([
    'và', 'là', 'của', 'có', 'được', 'trong', 'cho', 'với',
    'này', 'đó', 'các', 'những', 'một', 'đã', 'tôi', 'bạn', 'họ',
    'mà', 'về', 'theo', 'từ', 'hay', 'khi', 'thì', 'vì', 'nên',
    'để', 'nhưng', 'còn', 'cũng', 'lại', 'đây', 'rất', 'hơn',
    'như', 'thế', 'nào', 'ai', 'gì', 'sẽ', 'đến', 'ra', 'đi',
    'lên', 'xuống', 'vào', 'đang', 'bị', 'phải', 'nếu',
    'mình', 'ta', 'chúng', 'em', 'anh', 'chị', 'ông', 'bà'
])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)   # Loại URL
    text = re.sub(r'\d+', '', text)                  # Loại số
    text = re.sub(r'[^\w\s]', ' ', text)             # Loại ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()         # Chuẩn hóa khoảng trắng
    tokens = [t for t in text.split() if t not in VIETNAMESE_STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

# ==========================================
# 3. LUỒNG HUẤN LUYỆN MÔ HÌNH BACKEND
# ==========================================
@st.cache_resource
def train_model_backend():
    """Tải dataset UIT-VSFC và huấn luyện mô hình Logistic Regression"""
    try:
        dataset = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
        df_train = dataset['train'].to_pandas()
        
        df_train['processed'] = df_train['sentence'].apply(preprocess_text)
        
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)
        X_train = tfidf.fit_transform(df_train['processed'])
        y_train = df_train['sentiment']
        
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(X_train, y_train)
        
        return tfidf, model
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo hoặc huấn luyện mô hình học máy: {e}")
        return None, None

with st.spinner("⏳ Hệ thống đang khởi tạo mô hình AI và nạp dữ liệu huấn luyện... Vui lòng đợi trong giây lát."):
    tfidf, model = train_model_backend()

if tfidf and model:
    st.sidebar.success("✅ Hệ thống AI đã sẵn sàng phân tích!")
else:
    st.sidebar.error("❌ Không thể khởi tạo hệ thống AI.")

# ==========================================
# 4. LUỒNG PHÂN TÍCH SÂU TIÊU CỰC & ĐỀ XUẤT GIẢI PHÁP
# ==========================================
def analyze_negative_reason(text):
    """Hàm phân tích từ khóa trong câu tiêu cực để tìm ra nhóm nguyên nhân chính"""
    text_lower = text.lower()
    
    keywords = {
        "Phương pháp giảng dạy / Tốc độ": ["nhanh", "chậm", "khó hiểu", "chán", "buồn ngủ", "mơ hồ", "áp lực", "phương pháp", "giảng"],
        "Nội dung kiến thức / Bài tập": ["nhiều bài tập", "nặng", "khó", "lý thuyết", "slide", "giáo trình", "bài tập về nhà", "vô ích"],
        "Thái độ / Tương tác": ["gắt", "khó tính", "thiên vị", "không trả lời", "thái độ", "trừ điểm", "la", "mắng", "không nhiệt tình"],
        "Cơ sở vật chất / Thiết bị": ["mic", "loa", "bảng", "nóng", "phòng học", "chiếu", "máy chiếu", "mạng", "wifi"]
    }
    
    reasons = []
    for reason, kw_list in keywords.items():
        if any(kw in text_lower for kw in kw_list):
            reasons.append(reason)
            
    if not reasons:
        reasons.append("Phản hồi chung về môn học / Lý do khác")
        
    solutions = {
        "Phương pháp giảng dạy / Tốc độ": [
            "Điều chỉnh giảm bớt tốc độ giảng lý thuyết, tăng cường lấy ví dụ thực tế trực quan.",
            "Sử dụng các công cụ tương tác như Quizizz, Kahoot để khuấy động không khí lớp học.",
            "Dành 5-10 phút cuối giờ để tóm tắt lại các sơ đồ tư duy cốt lõi của bài học."
        ],
        "Nội dung kiến thức / Bài tập": [
            "Phân loại bài tập thành nhóm 'Bắt buộc (Cơ bản)' và 'Khuyến khích (Nâng cao)' để giảm tải áp lực.",
            "Cung cấp thêm video hướng dẫn giải bài tập mẫu hoặc tài liệu đọc thêm ngắn gọn.",
            "Xem xét lại độ dài và tính thực tế của Slide bài giảng, loại bỏ các phần lý thuyết quá hàn lâm."
        ],
        "Thái độ / Tương tác": [
            "Tạo một hòm thư góp ý ẩn danh hoặc buổi Q&A thoải mái để lắng nghe khó khăn của sinh viên.",
            "Kiềm chế cảm xúc trong lớp, thay đổi cách nhắc nhở bằng hình thức động viên.",
            "Tích cực phản hồi email hoặc tin nhắn thắc mắc của sinh viên về bài tập nhanh chóng hơn."
        ],
        "Cơ sở vật chất / Thiết bị": [
            "Báo cáo ngay với phòng ban quản trị thiết bị để đổi micro, sửa máy chiếu hoặc kiểm tra lại điều hòa.",
            "Chuẩn bị phương án dự phòng (ví dụ: gửi trước tài liệu bản mềm cho sinh viên đề phòng máy chiếu hỏng)."
        ],
        "Phản hồi chung về môn học / Lý do khác": [
            "Chủ động tổ chức một khảo sát ngắn giữa kỳ để sinh viên nói rõ hơn nguyện vọng cụ thể của mình.",
            "Thảo luận với các giảng viên cùng bộ môn để đổi mới cách tiếp cận môn học hấp dẫn hơn."
        ]
    }
    
    all_solutions = []
    for r in reasons:
        all_solutions.extend(solutions[r])
        
    return reasons, all_solutions

# ==========================================
# 5. GIAO DIỆN CHỨC NĂNG CỦA ỨNG DỤNG WEB
# ==========================================
tab1, tab2 = st.tabs(["📝 Phân tích từng câu một", "📁 Tải File phân tích hàng loạt"])

# ------------------------------------------
# TAB 1: NHẬP TỪNG CÂU
# ------------------------------------------
with tab1:
    st.header("Phân tích trực tiếp câu phản hồi")
    user_input = st.text_area("Nhập câu phản hồi của sinh viên tại đây:", height=100, placeholder="Ví dụ: Thầy dạy hơi nhanh và bài tập quá khó hiểu...")
    
    if st.button("Phân Tích Cảm Xúc 🔍", key="btn_single"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản trước khi bấm phân tích!")
        elif not model:
            st.error("Mô hình AI chưa được tải thành công.")
        else:
            processed = preprocess_text(user_input)
            vector = tfidf.transform([processed])
            pred = model.predict(vector)[0]
            
            labels_map = {0: "⚠️ Tiêu Cực (Negative)", 1: "😐 Trung Lập (Neutral)", 2: "✅ Tích Cực (Positive)"}
            colors_map = {0: "red", 1: "inverse", 2: "green"}
            
            st.markdown(f"#### Kết quả phân loại: :{colors_map[pred]}[{labels_map[pred]}]")
            
            if pred == 0:
                reasons, suggestions = analyze_negative_reason(user_input)
                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("🚨 **Vấn đề tiêu cực được phát hiện thuộc nhóm:**")
                    for r in reasons:
                        st.markdown(f"- 📌 *{r}*")
                with col2:
                    st.markdown("💡 **Gợi ý hướng giải quyết dành cho Giảng viên:**")
                    for s in suggestions:
                        st.markdown(f"- {s}")
            else:
                st.balloons()
                st.success("Tuyệt vời! Đây là một phản hồi mang tính xây dựng tích cực hoặc trung lập. Hãy tiếp tục phát huy!")

# ------------------------------------------
# TAB 2: TẢI FILE EXCEL / CSV
# ------------------------------------------
with tab2:
    st.header("Phân tích dữ liệu lớn từ File")
    
    uploaded_file = st.file_uploader("Chọn file dữ liệu của bạn (.csv, .xlsx)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_bytes = uploaded_file.getvalue()
                raw_lines = raw_bytes.decode('utf-8', errors='ignore').splitlines()
                sentences = [line.strip() for line in raw_lines if line.strip()]
                df_upload = pd.DataFrame({'sentence': sentences})
            else:
                df_excel = pd.read_excel(uploaded_file)
                combined_series = df_excel.astype(str).agg(lambda x: ' '.join([i for i in x if i.lower() != 'nan' and i.strip() != '']), axis=1)
                df_upload = pd.DataFrame({'sentence': combined_series})
            
            df_upload['sentence'] = df_upload['sentence'].apply(lambda x: re.sub(r'\bnan\b', '', str(x)).strip())
            df_upload['sentence'] = df_upload['sentence'].apply(lambda x: re.sub(r'\s+', ' ', str(x)))
            
            df_upload = df_upload[df_upload['sentence'].str.lower() != 'sentence']
            df_upload = df_upload[df_upload['sentence'].str.strip().str.len() > 0]
            
            st.success("🎯 Hệ thống đã chuẩn hóa và đưa dữ liệu phản hồi vào 1 cột duy nhất!")
            st.write("🔍 **5 dòng dữ liệu xem trước chuẩn cấu hình:**")
            st.dataframe(df_upload.head(5))
            
            if not model:
                st.error("Hệ thống AI chưa sẵn sàng.")
            else:
                if st.button("Bắt đầu phân tích hàng loạt 🚀", key="btn_bulk"):
                    with st.spinner("Hệ thống đang quét phân tích toàn bộ file..."):
                        
                        df_upload['processed_temp'] = df_upload['sentence'].apply(preprocess_text)
                        vectors_bulk = tfidf.transform(df_upload['processed_temp'])
                        preds_bulk = model.predict(vectors_bulk)
                        
                        sentiment_name_map = {0: "Tiêu Cực", 1: "Trung Lập", 2: "Tích Cực"}
                        df_upload['Kết quả cảm xúc'] = [sentiment_name_map[p] for p in preds_bulk]
                        
                        vấn_đề_list = []
                        giải_pháp_list = []
                        
                        for idx, row in df_upload.iterrows():
                            if row['Kết quả cảm xúc'] == "Tiêu Cực":
                                r_list, s_list = analyze_negative_reason(str(row['sentence']))
                                vấn_đề_list.append(", ".join(r_list))
                                giải_pháp_list.append(" | ".join(s_list))
                            else:
                                vấn_đề_list.append("—")
                                giải_pháp_list.append("—")
                                
                        df_upload['Vấn đề cụ thể'] = vấn_đề_list
                        df_upload['Gợi ý hướng giải quyết'] = giải_pháp_list
                        
                        df_upload = df_upload.drop(columns=['processed_temp'])
                        
                        # ---- HIỂN THỊ THỐNG KÊ TỔNG QUAN ----
                        st.write("---")
                        st.subheader("📊 Kết Quả Thống Kê Tổng Quan")
                        
                        counts = df_upload['Kết quả cảm xúc'].value_counts()
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("😊 Tổng số câu Tích Cực", f"{counts.get('Tích Cực', 0)} câu")
                        col_m2.metric("😐 Tổng số câu Trung Lập", f"{counts.get('Trung Lập', 0)} câu")
                        col_m3.metric("😞 Tổng số câu Tiêu Cực", f"{counts.get('Tiêu Cực', 0)} câu")
                        
                        # ✨ ĐÃ THAY ĐỔI TẠI ĐÂY: Chuyển biểu đồ cột sang BIỂU ĐỒ TRÒN THỂ HIỆN %
                        fig, ax = plt.subplots(figsize=(5, 5)) # Đặt khung vuông để biểu đồ tròn đều
                        colors = ['#22c55e' if x=='Tích Cực' else '#6b7280' if x=='Trung Lập' else '#ef4444' for x in counts.index]
                        
                        # Vẽ pie chart với định dạng % hiển thị là số thập phân 1 chữ số (ví dụ: 65.4%)
                        ax.pie(
                            counts.values, 
                            labels=counts.index, 
                            autopct='%1.1f%%', 
                            startangle=140, 
                            colors=colors,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
                        )
                        ax.set_title("Biểu đồ tỷ lệ phần trăm cảm xúc sinh viên trong File")
                        st.pyplot(fig)
                        
                        # ---- HIỂN THỊ BẢNG KẾT QUẢ ----
                        st.write("---")
                        st.subheader("📝 Bảng Dữ Liệu Sau Khi Phân Tích Chi Tiết")
                        st.dataframe(df_upload)
                        
                        @st.cache_data
                        def convert_df(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Phân tích cảm xúc')
                            return output.getvalue()
                            
                        xlsx_data = convert_df(df_upload)
                        st.download_button(
                            label="📥 Tải xuống File Excel kết quả (.xlsx)",
                            data=xlsx_data,
                            file_name='ket_qua_phan_tich_phan_hoi.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                        
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý dữ liệu file: {e}")