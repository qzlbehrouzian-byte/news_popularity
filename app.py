import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# تنظیمات اولیه صفحه و اعمال تم روشن، شاد و سفید به صورت مستقیم در کد
st.set_page_config(
    page_title="News Popularity Predictor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تزریق استایل‌های CSS برای سفید و شاد کردن محیط برنامه
st.markdown("""
    <style>
        /* پس‌زمینه اصلی سایت */
        .stApp {
            background-color: #ffffff;
            color: #1f2937;
        }
        /* رنگ دکمه اصلی */
        div.stButton > button:first-child {
            background-color: #4f46e5;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:first-child:hover {
            background-color: #4338ca;
            color: white;
        }
        /* رنگ متون کادرهای ورودی عدد */
        .stNumberInput label, .stSlider label, .stRadio label {
            color: #1f2937 !important;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ۱. لود داده‌ها و کش کردن آن‌ها برای سرعت بالاتر
@st.cache_data
def load_data():
    file_path = 'OnlineNewsPopularity.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"خطا در بارگذاری فایل داده‌ها: {e}")
    st.stop()

# آماده‌سازی متغیرها
features = ['num_imgs', 'is_weekend', 'n_tokens_content', 'global_sentiment_polarity']
X = df[features]
y_reg = df['shares']
y_cls = (y_reg > y_reg.median()).astype(int)

# آموزش مدل‌های پایه برای استفاده در اپلیکیشن
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_scaled, y_reg)

log_model = LogisticRegression()
log_model.fit(X_scaled, y_cls)

# طراحی رابط کاربری برنامه - وسط چین با متن تیره متناسب با تم سفید
st.markdown("""
    <div style="text-align: center; direction: rtl; margin-bottom: 20px;">
        <h1 style="color: #1f2937; font-family: Tahoma, Geneva, sans-serif;">سامانه هوشمند پیش‌بینی محبوبیت اخبار آنلاین</h1>
        <p style="font-size: 1.1em; color: #4b5563;">با تغییر ویژگی‌های زیر مشخص کنید مقاله شما چقدر در شبکه‌های اجتماعی به اشتراک گذاشته می‌شود.</p>
    </div>
""", unsafe_allow_html=True)

# ایجاد ستون‌ها برای دریافت ورودی
col1, col2 = st.columns(2)

with col1:
    num_imgs = st.number_input("تعداد تصاویر موجود در مقاله:", min_value=0, max_value=50, value=2)
    n_tokens_content = st.number_input("تعداد کلمات کل متن مقاله:", min_value=10, max_value=5000, value=500)
    
with col2:
    is_weekend_input = st.radio("آیا مقاله در آخر هفته منتشر می‌شود؟", ["خیر", "بله"])
    is_weekend = 1 if is_weekend_input == "بله" else 0
    global_sentiment_polarity = st.slider("لحن عاطفی مقاله از منفی تا مثبت:", min_value=-1.0, max_value=1.0, value=0.1)

# دکمه اجرای پیش‌بینی
if st.button("🚀 تحلیل و پیش‌بینی وضعیت مقاله"):
    # آماده‌سازی ورودی کاربر و استانداردسازی آن
    user_input = np.array([[num_imgs, is_weekend, n_tokens_content, global_sentiment_polarity]])
    user_input_scaled = scaler.transform(user_input)
    
    # اجرای پیش‌بینی‌ها
    pred_shares = ridge_model.predict(user_input_scaled)[0]
    pred_shares = max(0, int(pred_shares))
    
    pred_class = log_model.predict(user_input_scaled)[0]
    pred_proba = log_model.predict_proba(user_input_scaled)[0][1]
    
    st.markdown("""<hr style="border-top: 2px dashed #cca;">""", unsafe_allow_html=True)
    
    # تیتر بخش نتایج به صورت راست به چپ
    st.markdown('<div style="text-align: right; direction: rtl; color: #1f2937;"><h3>🎯 نتایج ارزیابی مدل‌ها</h3></div>', unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.info(f"**پیش‌بینی تعداد اشتراک‌گذاری (رگرسیون ریج):** {pred_shares} مرتبه")
    
    with res_col2:
        status = "محبوب و پرمخاطب" if pred_class == 1 else "معمولی و کم‌مخاطب"
        st.success(f"**وضعیت مقاله (رگرسیون لجستیک):** {status} با احتمال {pred_proba:.2%}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- رسم نمودار آماری با تم روشن و هماهنگ ---
    st.markdown('<div style="text-align: right; direction: rtl; color: #1f2937;"><h4>📊 موقعیت مقاله شما در توزیع کل داده‌ها</h4></div>', unsafe_allow_html=True)
    
    # تنظیم استایل روشن برای نمودار متناسب با تم جدید
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    # رسم نمودار توزیع
    upper_limit = np.percentile(df['shares'], 95)
    filtered_shares = df['shares'][df['shares'] <= upper_limit]
    
    sns.histplot(filtered_shares, kde=True, color="#4f46e5", ax=ax, bins=50, stat="density", alpha=0.5)
    
    # رسم خط قرمز رنگ پیش‌بینی کاربر
    ax.axvline(pred_shares, color="#dc2626", linestyle="--", linewidth=2.5, label=f'Your Article Prediction ({pred_shares})')
    
    # زیباسازی نمودار (متون تیره برای بک‌گراند سفید)
    ax.set_title("Distribution of Article Shares in Dataset vs. Your Prediction", fontsize=12, pad=10, color="#1f2937")
    ax.set_xlabel("Number of Shares", fontsize=10, color="#1f2937")
    ax.set_ylabel("Density", fontsize=10, color="#1f2937")
    ax.tick_params(colors='#1f2937')
    ax.legend(loc="upper right", facecolor='#ffffff', edgecolor='#e5e7eb')
    ax.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')
    
    # نمایش نمودار در وب‌اپلیکیشن
    st.pyplot(fig)