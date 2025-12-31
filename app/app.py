import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide", page_icon="😴")

st.title("😴 Dự đoán & Cải thiện Chất lượng Giấc ngủ")

# --- KHAI BÁO BIẾN TOÀN CỤC ---
CAT_COLS = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

# --- HÀM TẠO GỢI Ý (MỚI) ---
def get_recommendations(duration, stress, physical, bmi, disorder, quality_pred):
    tips = []
    
    # 1. Lời khuyên dựa trên kết quả dự đoán (AI Predicted)
    if quality_pred == 0: # Kém
        tips.append("⚠️ **Cảnh báo chung:** AI dự đoán chất lượng giấc ngủ của bạn KÉM. Cần rà soát lại lối sống.")
    elif quality_pred == 1: # Trung bình
        tips.append("ℹ️ **Lưu ý:** Giấc ngủ ở mức chấp nhận được, hãy cố gắng cải thiện thêm.")
    
    # 2. Lời khuyên về Rối loạn giấc ngủ (Input User)
    # SỬA LỖI TẠI ĐÂY: Chuyển hết về chữ thường để so sánh
    if disorder.lower() != "none":
        tips.append(f"🏥 **Bệnh lý:** Bạn đã khai báo có **{disorder}**. Hãy tuân thủ phác đồ điều trị của bác sĩ.")
    else:
        # Nếu không có bệnh lý (None) nhưng AI vẫn dự báo ngủ Kém (0)
        if quality_pred == 0:
             tips.append("🛌 **Môi trường ngủ:** Bạn không có bệnh lý nền, nhưng giấc ngủ vẫn kém. Hãy kiểm tra: nhiệt độ phòng, ánh sáng, tiếng ồn hoặc nệm gối.")

    # 3. Lời khuyên dựa trên Thời lượng ngủ
    if duration < 6.0:
        tips.append("⏰ **Thời lượng:** Bạn ngủ quá ít (< 6h). Hãy cố gắng ngủ sớm hơn.")
    elif duration > 9.0:
        tips.append("⏰ **Thời lượng:** Ngủ nướng quá nhiều cũng gây mệt mỏi.")

    # 4. Lời khuyên dựa trên Stress
    if stress > 6:
        tips.append("🤯 **Căng thẳng:** Mức Stress cao là nguyên nhân chính. Hãy thử: Thiền, đọc sách giấy, hạn chế tin tức tiêu cực.")

    # 5. Lời khuyên dựa trên Vận động
    if physical < 30:
        tips.append("🏃 **Vận động:** Tăng cường đi bộ hoặc tập nhẹ 30p/ngày để cơ thể dễ chìm vào giấc ngủ.")
    
    # 6. Lời khuyên dựa trên BMI
    if bmi in ['Overweight', 'Obese']:
        tips.append("⚖️ **Cân nặng:** Thừa cân có thể gây chèn ép đường thở khi nằm. Giảm cân sẽ giúp cải thiện đáng kể.")

    return tips

# --- LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    except FileNotFoundError:
        return None, None, None, None
    
    if 'Person ID' in df.columns:
        df.set_index("Person ID", inplace=True)

    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("none")
    df['Blood Pressure'] = df['Blood Pressure'].astype(str)
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

    def label_quality(x):
        if x <= 5: return 0 
        elif x <= 7: return 1 
        else: return 2 
    
    df['SleepQualityLabel'] = df['Quality of Sleep'].apply(label_quality)

    le_dict = {}
    df_encoded = df.copy()
    
    for col in CAT_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df_encoded.drop(['Quality of Sleep', 'SleepQualityLabel', 'Daily Steps'], axis=1)
    y = df_encoded['SleepQualityLabel']

    return df, X, y, le_dict

df_original, X, y, le_dict = load_and_process_data()

if df_original is None:
    st.error("Không tìm thấy file dữ liệu.")
    st.stop()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# --- SIDEBAR: CẤU HÌNH ---
st.sidebar.header("⚙️ Cấu hình")
model_option = st.sidebar.selectbox("Mô hình:", ["Decision Tree", "Random Forest"])

if model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Độ sâu (Max Depth)", 1, 20, 3) 
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
else:
    n_estimators = st.sidebar.slider("Số lượng cây", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)

st.sidebar.markdown("---")
st.sidebar.success(f"Độ chính xác: **{acc:.2%}**")

# --- GIAO DIỆN CHÍNH ---
tab1, tab2 = st.tabs(["🔮 Dự đoán & Lời khuyên", "📈 Đánh giá mô hình"])

# TAB 1: DỰ ĐOÁN
with tab1:
    st.markdown("#### Nhập thông tin sức khỏe")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Giới tính", le_dict['Gender'].classes_)
            # Cho phép nhập tuổi rộng hơn
            age = st.number_input("Tuổi", min_value=10, max_value=90, value=30)
            occupation = st.selectbox("Nghề nghiệp", le_dict['Occupation'].classes_)
            
        with col2:
            sleep_duration = st.number_input("Thời lượng ngủ (giờ)", 3.0, 12.0, 7.0, step=0.1)
            physical = st.slider("Hoạt động thể chất (phút/ngày)", 0, 120, 40)
            stress = st.slider("Mức độ Stress (1-10)", 1, 10, 5)
            
        with col3:
            bmi = st.selectbox("BMI Category", le_dict['BMI Category'].classes_)
            heart_rate = st.number_input("Nhịp tim (bpm)", 40, 120, 70)
            sleep_disorder = st.selectbox("Rối loạn giấc ngủ", le_dict['Sleep Disorder'].classes_)

        col4, col5 = st.columns(2)
        with col4:
            sys_bp = st.number_input("Huyết áp tâm thu", 80, 200, 120)
        with col5:
            dia_bp = st.number_input("Huyết áp tâm trương", 50, 130, 80)

        submit_btn = st.form_submit_button("Xem kết quả & Lời khuyên")

    if submit_btn:
        # Tạo input dataframe
        input_data = pd.DataFrame({
            'Gender': [gender], 'Age': [age], 'Occupation': [occupation],
            'Sleep Duration': [sleep_duration], 'Physical Activity Level': [physical],
            'Stress Level': [stress], 'BMI Category': [bmi],
            'Heart Rate': [heart_rate], 'Sleep Disorder': [sleep_disorder],
            'Systolic': [sys_bp], 'Diastolic': [dia_bp]
        })
        
        # Mã hóa input
        for col in CAT_COLS:
            input_data[col] = le_dict[col].transform(input_data[col])
            
        try:
            pred = model.predict(input_data)[0]
            
            # Mapping kết quả
            result_map = {0: "Kém (Poor)", 1: "Trung bình (Normal)", 2: "Tốt (Good)"}
            
            st.divider()
            
            # Hiển thị kết quả chính
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.subheader("Kết quả:")
                if pred == 0:
                    st.error(f"🔴 {result_map[pred]}")
                elif pred == 1:
                    st.warning(f"🟠 {result_map[pred]}")
                else:
                    st.success(f"🟢 {result_map[pred]}")
            
            with col_res2:
                # Gọi hàm lấy lời khuyên
                recommendations = get_recommendations(sleep_duration, stress, physical, bmi, sleep_disorder, pred)
                
                st.subheader("💡 Gợi ý cải thiện:")
                if len(recommendations) > 0:
                    for tip in recommendations:
                        st.info(tip)
                else:
                    st.success("🎉 Bạn đang duy trì lối sống rất tốt! Hãy tiếp tục phát huy.")
                    
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

# TAB 2: LEARNING CURVE
with tab2:
    st.subheader("Learning Curve")
    if st.button("Vẽ biểu đồ"):
        with st.spinner('Đang xử lý...'):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, scoring='accuracy', 
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42
            )
            
            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_sizes, train_mean, label="Training Score", marker='o')
            ax.plot(train_sizes, val_mean, label="Validation Score", marker='o')
            ax.set_xlabel("Số lượng mẫu")
            ax.set_ylabel("Độ chính xác")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
    else:
        st.info("Nhấn nút trên để hiển thị biểu đồ.")