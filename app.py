import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# تنظیمات اولیه صفحه استریم‌لیت
st.set_page_config(page_title="News Popularity Predictor", layout="wide")

# ۱. لود داده‌ها و کش کردن آن‌ها برای سرعت بالاتر
@st.cache_data
def load_data():
    # مسیر فایل به صورت محلی تنظیم شده تا در کنار فایل کد قرار گیرد
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

# طراحی رابط کاربری برنامه
st.title("سامانه هوشمند پیش‌بینی محبوبیت اخبار آنلاین")
st.markdown("با تغییر ویژگی‌های زیر مشخص کنید مقاله شما چقدر در شبکه‌های اجتماعی به اشتراک گذاشته می‌شود.")

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
    pred_class = log_model.predict(user_input_scaled)[0]
    pred_proba = log_model.predict_proba(user_input_scaled)[0][1]
    
    st.markdown("---\")
    st.subheader("🎯 نتایج ارزیابی مدل‌ها")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.info(f"**پیش‌بینی تعداد اشتراک‌گذاری رگرسیون ریج:** {int(pred_shares)} مرتبه")
    
    with res_col2:
        status = "محبوب و پرمخاطب" if pred_class == 1 else "معمولی و کم‌مخاطب"
        st.success(f"**وضعیت مقاله رگرسیون لجستیک:** {status} با احتمال {pred_proba:.2%}")