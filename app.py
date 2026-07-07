import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# تنظیمات اولیه صفحه
st.set_page_config(
    page_title="News Popularity Predictor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تزریق استایل‌های CSS برای تم سفید خالص، نوشته‌های مشکی و کادرهای پاستلی روشن
st.markdown("""
    <style>
        /* پس‌زمینه اصلی سایت و رنگ متون پایه */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* مشکی کردن تمام متون، لیبل‌ها و تیترهای استریم‌لیت */
        h1, h2, h3, h4, h5, h6, p, label, .stRadio p {
            color: #000000 !important;
            font-weight: bold !important;
        }
        
        /* استایل اختصاصی کادر اول: تعداد تصاویر (آبی پاستلی بسیار روشن) */
        div[data-testid="stBlock"] [data-testid="element-container"]:nth-of-type(1) div[data-baseweb="input"] {
            background-color: #e0f2fe !important;
            border: 1px solid #7dd3fc !important;
            border-radius: 8px !important;
        }
        
        /* استایل اختصاصی کادر دوم: تعداد کلمات (سبز پاستلی بسیار روشن) */
        div[data-testid="stBlock"] [data-testid="element-container"]:nth-of-type(2) div[data-baseweb="input"] {
            background-color: #dcfce7 !important;
            border: 1px solid #86efac !important;
            border-radius: 8px !important;
        }
        
        /* مشکی و ضخیم کردن عدد داخل کادرهای ورودی */
        div[data-baseweb="input"] input {
            color: #000000 !important;
            font-weight: bold !important;
            font-size: 1.05em !important;
        }
        
        /* استایل اختصاصی دکمه اصلی (رنگ شاد و متن سفید برای خوانایی) */
        div.stButton > button:first-child {
            background-color: #4f46e5 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: none !important;
            font-size: 1.1em !important;
            padding: 10px 24px !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #4338ca !important;
        }
        
        /* تغییر ظاهر کادرهای پیش‌فرض ارور یا اعلان استریم‌لیت */
        div.stAlert {
            background-color: #f1f5f9 !important;
            border: 1px solid #e2e8f0 !important;
        }
        div[data-testid="stNotificationV2"] {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ۱. لود داده‌ها و کش کردن آن‌ها
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

# آموزش مدل‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_scaled, y_reg)

log_model = LogisticRegression()
log_model.fit(X_scaled, y_cls)

# طراحی رابط کاربری برنامه - کاملاً مشکی و راست‌به‌چپ
st.markdown("""
    <div style="text-align: center; direction: rtl; margin-bottom: 30px;">
        <h1 style="color: #000000; font-family: Tahoma, Geneva, sans-serif; font-size: 2.3em;">سامانه هوشمند پیش‌بینی محبوبیت اخبار آنلاین</h1>
        <p style="font-size: 1.2em; color: #334155;">با تغییر ویژگی‌های زیر مشخص کنید مقاله شما چقدر در شبکه‌های اجتماعی به اشتراک گذاشته می‌شود.</p>
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

st.markdown("<br>", unsafe_allow_html=True)

# دکمه اجرای پیش‌بینی
if st.button("🚀 تحلیل و پیش‌بینی وضعیت مقاله"):
    user_input = np.array([[num_imgs, is_weekend, n_tokens_content, global_sentiment_polarity]])
    user_input_scaled = scaler.transform(user_input)
    
    pred_shares = ridge_model.predict(user_input_scaled)[0]
    pred_shares = max(0, int(pred_shares))
    
    pred_class = log_model.predict(user_input_scaled)[0]
    pred_proba = log_model.predict_proba(user_input_scaled)[0][1]
    
    st.markdown("""<hr style="border-top: 2px dashed #cbd5e1;">""", unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: right; direction: rtl;"><h2>🎯 نتایج ارزیابی مدل‌ها</h2></div>', unsafe_allow_html=True)
    
    # طراحی باکس‌های خروجی سفارشی با پس‌زمینه فوق‌العاده روشن و متن مشکی خالص
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
            <div style="background-color: #eff6ff; border-right: 5px solid #3b82f6; padding: 15px; border-radius: 8px; text-align: right; direction: rtl;">
                <span style="color: #000000; font-weight: bold; font-size: 1.1em;">پیش‌بینی تعداد اشتراک‌گذاری (رگرسیون ریج):</span>
                <span style="color: #1e40af; font-weight: bold; font-size: 1.2em; margin-right: 5px;">{pred_shares} مرتبه</span>
            </div>
        """, unsafe_allow_html=True)
    
    with res_col2:
        status = "محبوب و پرمخاطب" if pred_class == 1 else "معمولی و کم‌مخاطب"
        box_color = "#f0fdf4" if pred_class == 1 else "#fff7ed"
        border_color = "#22c55e" if pred_class == 1 else "#f97316"
        text_color = "#166534" if pred_class == 1 else "#9a3412"
        
        st.markdown(f"""
            <div style="background-color: {box_color}; border-right: 5px solid {border_color}; padding: 15px; border-radius: 8px; text-align: right; direction: rtl;">
                <span style="color: #000000; font-weight: bold; font-size: 1.1em;">وضعیت مقاله (رگرسیون لجستیک):</span>
                <span style="color: {text_color}; font-weight: bold; font-size: 1.1em; margin-right: 5px;">{status} با احتمال {pred_proba:.2%}</span>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- نمودار آماری با تم کاملاً روشن و هماهنگ ---
    st.markdown('<div style="text-align: right; direction: rtl;"><h3>📊 موقعیت مقاله شما در توزیع کل داده‌ها</h3></div>', unsafe_allow_html=True)
    
    # ست کردن استایل سفید برای پلات
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    upper_limit = np.percentile(df['shares'], 95)
    filtered_shares = df['shares'][df['shares'] <= upper_limit]
    
    # رسم هیستوگرام با رنگ بنفش/آبی شاد
    sns.histplot(filtered_shares, kde=True, color="#4f46e5", ax=ax, bins=50, stat="density", alpha=0.4)
    
    # خط قرمز پیش‌بینی
    ax.axvline(pred_shares, color="#dc2626", linestyle="--", linewidth=2.5, label=f'Your Prediction ({pred_shares})')
    
    # مشکی کردن کامل متون و اجزای نمودار برای هماهنگی با تم
    ax.set_title("Distribution of Article Shares in Dataset vs. Your Prediction", fontsize=11, pad=10, color="#000000", fontweight='bold')
    ax.set_xlabel("Number of Shares", fontsize=10, color="#000000", fontweight='bold')
    ax.set_ylabel("Density", fontsize=10, color="#000000", fontweight='bold')
    ax.tick_params(colors='#000000', labelsize=9)
    ax.legend(loc="upper right", facecolor='#ffffff', edgecolor='#cbd5e1')
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    
    st.pyplot(fig)