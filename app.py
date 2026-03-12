import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os
from math import radians, sin, cos, sqrt, atan2
import openai

# ---------- 页面配置 ----------
st.set_page_config(page_title="助贷导流预测系统", layout="centered")

# ---------- 注入自定义 CSS ----------


st.markdown("""
<style>

    /* 整体背景与主卡片效果 */
    .stApp {
        background-color: #f5f7fa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .main > div {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem 1rem 2rem 1rem;
    }
    /* 模拟 resume-card 的白色卡片 */
    .block-container {
        background-color: #ffffff;
        border-radius: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.03), 0 2px 8px rgba(0, 20, 40, 0.02);
        padding: 2rem 2.5rem !important;
    }
    /* 标题区样式 */
    .custom-header h1 {
        font-size: 28px;
        font-weight: 600;
        color: #0b1c2e;
        margin-bottom: 8px;
    }
    .custom-header .desc {
        font-size: 15px;
        color: #5f6c7a;
        border-left: 3px solid #0066cc;
        padding: 10px 16px;
        background: #f8faff;
        border-radius: 0 8px 8px 0;
        margin-bottom: 32px;
    }
    /* 区块标题 (带数字圈) */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1e2f40;
        margin: 24px 0 16px 0;
        display: flex;
        align-items: center;
    }
    .section-title span {
        background-color: #ecf2ff;
        color: #0066cc;
        font-size: 15px;
        font-weight: 500;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        margin-right: 10px;
    }
    /* 统一输入框样式 */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 10px 16px !important;
        font-size: 15px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        transition: all 0.15s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #0066cc !important;
        box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
    }
    /* 下拉箭头美化 */
    .stSelectbox select {
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%235f6c7a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 16px center;
        background-size: 16px;
    }
    /* 按钮样式 */
    .stButton button {
        border-radius: 40px !important;
        padding: 8px 28px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        border: 1px solid #d0dbe8 !important;
        background: white !important;
        color: #2e405b !important;
        transition: all 0.15s;
    }
    .stButton button:hover {
        background: #f0f4f9 !important;
        border-color: #b4c1d1 !important;
    }
    /* 主按钮 (用于预测和最佳决策) */
    .stButton button[kind="primary"] {
        background: #0066cc !important;
        border: 1px solid #0066cc !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,102,204,0.15);
    }
    .stButton button[kind="primary"]:hover {
        background: #004999 !important;
        border-color: #004999 !important;
    }
    /* 指标卡片微调 */
    .stMetric {
        background: #f8faff;
        border-radius: 16px;
        padding: 16px;
        border-left: 3px solid #0066cc;
    }
    /* 数据表格边框柔和 */
    .stDataFrame {
        border: 1px solid #edf2f7;
        border-radius: 16px;
        overflow: hidden;
    }
    /* 小提示文字 */
    .small-note {
        font-size: 13px;
        color: #97a6b9;
        text-align: right;
        margin-top: 8px;
    }
    
    .section-desc {
    font-size: 15px;
    color: #3d4e63;
    margin: 8px 0 20px 0;
    padding: 0 0 0 12px;
    border-left: 3px solid #cbd5e0;
    background: #f9fbfd;
    border-radius: 0 8px 8px 0;
    line-height: 1.5;
}
    .stDataFrame td {
    text-align: left !important;
}
            
</style>
""", unsafe_allow_html=True)

# ---------- 标题区 ----------
st.markdown("""
<div class="custom-header">
    <h1>助贷导流 · 智能预测</h1>
</div>
""", unsafe_allow_html=True)

# ---------- 加载模型和组件 ----------
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data_now')

    models = {
        'XGBoost (平衡)': joblib.load(os.path.join(data_path, 'xgb_balanced.pkl')),
        '随机森林 (平衡)': joblib.load(os.path.join(data_path, 'rf_balanced.pkl')),
        '逻辑回归 (改进)': joblib.load(os.path.join(data_path, 'lr_improved.pkl')),
    }
    scaler = joblib.load(os.path.join(data_path, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(data_path, 'feature_names.pkl'))
    label_encoders = joblib.load(os.path.join(data_path, 'label_encoders.pkl'))
    with open(os.path.join(data_path, 'label_mappings.json'), 'r', encoding='utf-8') as f:
        label_mappings = json.load(f)
    return models, scaler, feature_names, label_encoders, label_mappings

models, scaler, FEATURE_NAMES, label_encoders, label_mappings = load_models()

# ---------- DeepSeek 客户端配置（兼容新版和旧版 openai）----------
deepseek_available = False
try:
    if "DEEPSEEK_API_KEY" in st.secrets:
        openai.api_key = st.secrets["DEEPSEEK_API_KEY"]
        openai.api_base = "https://api.deepseek.com/v1"  # 旧版使用 api_base
        deepseek_available = True
    else:
        st.warning("未配置 DeepSeek API 密钥，解释功能将不可用。")
except FileNotFoundError:
    st.warning("未找到 secrets.toml 配置文件，解释功能将不可用。请创建 .streamlit/secrets.toml 并添加 DEEPSEEK_API_KEY。")
except Exception as e:
    st.warning(f"读取 secrets 时出错: {e}，解释功能将不可用。")

# ---------- 定义数值列（用于标准化） ----------
NUM_COLS = ['amount', 'idInfo.birthDate', 'idInfo.validityDate', 'pictureInfo.0.faceScore',
            'deviceInfo.gpsLatitude', 'deviceInfo.gpsLongitude', 'term',
            'company_name_len', 'distance_to_capital']
NUM_COLS = [col for col in NUM_COLS if col in FEATURE_NAMES]

# ---------- 辅助函数 ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# 省会城市坐标
CAPITAL_COORDS = {
    '北京市': (39.9042, 116.4074), '天津市': (39.3434, 117.3616),
    '上海市': (31.2304, 121.4737), '重庆市': (29.4316, 106.9123),
    '河北省': (38.0428, 114.5149), '山西省': (37.8735, 112.5624),
    '辽宁省': (41.8057, 123.4315), '吉林省': (43.8965, 125.3261),
    '黑龙江省': (45.7421, 126.6579), '江苏省': (32.0603, 118.7969),
    '浙江省': (30.2741, 120.1551), '安徽省': (31.8206, 117.2272),
    '福建省': (26.0745, 119.2965), '江西省': (28.6829, 115.8582),
    '山东省': (36.6712, 116.9911), '河南省': (34.7466, 113.6253),
    '湖北省': (30.5928, 114.3055), '湖南省': (28.1127, 112.9836),
    '广东省': (23.1321, 113.2679), '海南省': (20.0174, 110.3493),
    '四川省': (30.5723, 104.0668), '贵州省': (26.5982, 106.7074),
    '云南省': (25.0459, 102.7100), '陕西省': (34.3416, 108.9398),
    '甘肃省': (36.0596, 103.8263), '青海省': (36.6207, 101.7782),
    '台湾省': (25.0380, 121.5645), '内蒙古自治区': (40.8174, 111.7653),
    '广西壮族自治区': (22.8167, 108.3669), '西藏自治区': (29.6469, 91.1172),
    '宁夏回族自治区': (38.4712, 106.2587), '新疆维吾尔自治区': (43.8256, 87.6168),
    '香港特别行政区': (22.3080, 114.1717), '澳门特别行政区': (22.1905, 113.5479),
}

def correct_province(prov_str):
    if pd.isna(prov_str) or prov_str == '':
        return np.nan
    prov_str = str(prov_str).strip()
    keyword_map = {
        '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
        '河北': '河北省', '山西': '山西省', '内蒙古': '内蒙古自治区',
        '辽宁': '辽宁省', '吉林': '吉林省', '黑龙江': '黑龙江省',
        '江苏': '江苏省', '浙江': '浙江省', '安徽': '安徽省',
        '福建': '福建省', '江西': '江西省', '山东': '山东省',
        '河南': '河南省', '湖北': '湖北省', '湖南': '湖南省',
        '广东': '广东省', '广西': '广西壮族自治区', '海南': '海南省',
        '四川': '四川省', '贵州': '贵州省', '云南': '云南省',
        '西藏': '西藏自治区', '陕西': '陕西省', '甘肃': '甘肃省',
        '青海': '青海省', '宁夏': '宁夏回族自治区', '新疆': '新疆维吾尔自治区',
        '香港': '香港特别行政区', '澳门': '澳门特别行政区', '台湾': '台湾省',
    }
    for kw, std in keyword_map.items():
        if kw in prov_str:
            return std
    return np.nan



# ---------- 特征工程函数 ----------
def preprocess_input(input_dict):
    input_df = pd.DataFrame([input_dict])









    input_df['company_name_len'] = input_df['companyInfo.companyName'].astype(str).apply(len)
    input_df['is_limited'] = input_df['companyInfo.companyName'].astype(str).str.contains('有限公司|有限责任公司', na=False).astype(int)
    input_df['is_individual'] = input_df['companyInfo.companyName'].astype(str).str.contains('个体经营|个人工作室', na=False).astype(int)
    input_df['companyName_missing'] = input_df['companyInfo.companyName'].isnull().astype(int)
    input_df.drop('companyInfo.companyName', axis=1, inplace=True)

    input_df['idInfo.birthDate'] = input_df['idInfo.birthDate'].clip(18, 100)
    input_df['pictureInfo.0.faceScore'] = input_df['pictureInfo.0.faceScore'].clip(0, 100)

    missing_cols = [col for col in FEATURE_NAMES if col.endswith('_missing')]
    for col in missing_cols:
        input_df[col] = 0

    input_df['gender'] = input_df['idInfo.gender'].map({'M': 0, 'F': 1})
    input_df.drop('idInfo.gender', axis=1, inplace=True)

    input_df['nation_group'] = input_df['idInfo.nation'].apply(lambda x: 0 if x == '汉' else 1)
    input_df.drop('idInfo.nation', axis=1, inplace=True)

    input_df['is_cross_domain'] = input_df['deviceInfo.isCrossDomain'].map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
    input_df.drop('deviceInfo.isCrossDomain', axis=1, inplace=True)

    bins = [18, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    input_df['age_group'] = pd.cut(input_df['idInfo.birthDate'], bins=bins, labels=labels, right=True)

    def get_capital_coords(prov):
        if pd.isna(prov):
            return (np.nan, np.nan)
        prov_str = str(prov).strip()
        return CAPITAL_COORDS.get(prov_str, (np.nan, np.nan))

    coords = input_df['province'].apply(get_capital_coords)
    input_df['capital_lat'] = coords.apply(lambda x: x[0])
    input_df['capital_lon'] = coords.apply(lambda x: x[1])
    input_df['distance_to_capital'] = input_df.apply(
        lambda row: haversine(row['deviceInfo.gpsLatitude'], row['deviceInfo.gpsLongitude'],
                              row['capital_lat'], row['capital_lon']) if not pd.isna(row['capital_lat']) else np.nan, axis=1
    )
    input_df['distance_to_capital'].fillna(500, inplace=True)
    input_df['in_capital_city'] = (input_df['distance_to_capital'] < 50).astype(int)
    input_df.drop(['capital_lat', 'capital_lon'], axis=1, inplace=True)

    for col, le in label_encoders.items():
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            try:
                input_df[col] = le.transform([val])[0]
            except:
                input_df[col] = 0

    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURE_NAMES]



    # # 在 preprocess_input 函数返回前添加：
    # for col in input_df.columns:
    #     # 如果列的类型是 object，尝试转换为数值，无法转换的填充 0
    #     if input_df[col].dtype == 'object':
    #         input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    # # 确保所有列最终为浮点数（模型通常接受浮点数）
    # input_df[col] = input_df[col].astype(float)






    return input_df

# ---------- 初始化 session_state ----------
if 'last_input_df' not in st.session_state:
    st.session_state.last_input_df = None
if 'last_pred_proba' not in st.session_state:
    st.session_state.last_pred_proba = None
if 'last_pred_class' not in st.session_state:
    st.session_state.last_pred_class = None
if 'last_shap_fig' not in st.session_state:
    st.session_state.last_shap_fig = None
if 'last_model_choice' not in st.session_state:
    st.session_state.last_model_choice = 'XGBoost (平衡)'
if 'pred_explanation' not in st.session_state:
    st.session_state.pred_explanation = None
if 'best_reason' not in st.session_state:
    st.session_state.best_reason = None
if 'last_user_advantages' not in st.session_state:   # 新增：保存用户优势特征
    st.session_state.last_user_advantages = ""

# ---------- 生成预测解释的 API 函数（增加 max_tokens）----------
def generate_prediction_explanation(proba, pos_features, neg_features, model_choice):
    if not deepseek_available:
        return "（API未配置，无法生成解释）"
    # 格式化正负特征，包含值和影响方向
    pos_desc = "、".join([f"{name}为{val:.1f}（提升{shap:.2f}）" for name, val, shap in pos_features[:3]])
    neg_desc = "、".join([f"{name}为{val:.1f}（降低{-shap:.2f}）" for name, val, shap in neg_features[:3]])
    prompt = f"""你是一位信贷审核专家，请向用户简单解释为什么他的贷款申请通过概率为{proba:.1%}。
主要有利因素：{pos_desc}；主要不利因素：{neg_desc}。请结合这些数值说明它们如何影响结果。条理清晰，层次分明，字数200。"""
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400          # 增加输出长度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"解释生成失败：{e}"

# ---------- 生成最佳决策理由的 API 函数（增加 max_tokens）----------
def generate_best_decision_reason(best_company, best_prob, other_companies_top3, user_advantages):
    if not deepseek_available:
        return "（API未配置，无法生成解释）"
    others = "、".join([f"{name}({prob:.1%})" for name, prob in other_companies_top3])
    prompt = f"""你是一位信贷顾问，请向用户简单解释为什么{best_company}是当前用户的最佳选择（通过概率{best_prob:.1%}），而其他公司如{others}稍低。
用户的优势特征有：{user_advantages}。请结合这些特征说明该公司为何更适合用户。条理清晰，层次分明，字数200。"""
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400          # 增加输出长度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"理由生成失败：{e}"
    

# ========== 第一部分：个人资料（无表单） ==========
st.markdown('<div class="section-title"><span>①</span> 个人资料</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    partner_code = st.selectbox("合作方代码  ", options=label_encoders['partner_code'].classes_, key='partner_code')
    amount = st.number_input("申请金额  ", min_value=0, value=100000, step=1000, key='amount')
    bankCode = st.number_input("银行代码  ", min_value=18, max_value=403, value=105, step=1, key='bankCode')
    company_name = st.text_input("公司名称  ", value="某某有限公司", key='company_name')
    degree = st.number_input("学历等级  ", min_value=1, max_value=5, value=3, step=1, key='degree')
    age = st.number_input("年龄  ", min_value=18, max_value=100, value=35, step=1, key='age')
    gender = st.selectbox("性别  ", options=['M', 'F'], key='gender')
    nation = st.text_input("民族  ", value="汉", key='nation')
    link0 = st.selectbox("第一联系人关系  ", options=label_encoders['linkmanList.0.relationship'].classes_, key='link0')
    link1 = st.selectbox("第二联系人关系  ", options=label_encoders['linkmanList.1.relationship'].classes_, key='link1')
    marital = st.number_input("婚姻状况编码  ", min_value=1, max_value=4, value=1, step=1, key='marital')
    purpose = st.selectbox("贷款用途  ", options=label_encoders['purpose'].classes_, key='purpose')
    term = st.number_input("贷款期限(月)  ", value=12, step=1, key='term')
    face_score = st.number_input("人脸评分  ", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key='face_score')

with col2:
    city = st.text_input("城市名称 (仅供展示)", value="某某城市", key='city')
    industry = st.selectbox("行业代码  ", options=label_encoders['companyInfo.industry'].classes_, key='industry')
    occupation = st.number_input("职业代码  ", min_value=11, max_value=90, value=24, step=1, key='occupation')
    customer_source = st.selectbox("客户来源  ", options=label_encoders['customerSource'].classes_, key='customer_source')
    validity_date = st.number_input("身份证有效期(剩余天数)  ", value=3650, step=100, key='validity_date')
    income = st.number_input("收入等级  ", min_value=1, max_value=5, value=3, step=1, key='income')
    job_func = st.number_input("工作职能编码  ", min_value=1, max_value=10, value=1, step=1, key='job_func')
    province = st.selectbox("省份  ", options=label_encoders['province'].classes_, key='province')
    reside = st.number_input("居住情况编码  ", min_value=1, max_value=5, value=1, step=1, key='reside')
    lat = st.number_input("GPS纬度  ", value=23.1321, format="%.6f", key='lat')
    lon = st.number_input("GPS经度  ", value=113.2679, format="%.6f", key='lon')
    os_type = st.selectbox("操作系统  ", options=label_encoders['deviceInfo.osType'].classes_, key='os_type')
    is_cross_domain = st.selectbox("是否跨域  ", options=['FALSE', 'TRUE'], key='is_cross_domain')

st.markdown("---")

# ========== 第二部分：模型选择与预测 ==========
st.markdown('<div class="section-title"><span>②</span> 模型选择与预测</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">选择机器学习模型，点击“预测”查看通过概率及影响因素分析（SHAP值）。</div>', unsafe_allow_html=True)

model_choice = st.selectbox("选择预测模型", options=list(models.keys()), key='model_choice')

if st.button(" 预测", use_container_width=True, type="primary"):
    # 收集当前所有输入值
    input_dict = {
        'partner_code': partner_code,
        'amount': amount,
        'bankCardInfo.bankCode': bankCode,
        'companyInfo.industry': industry,
        'companyInfo.occupation': occupation,
        'customerSource': customer_source,
        'degree': degree,
        'idInfo.birthDate': age,
        'idInfo.validityDate': validity_date,
        'income': income,
        'jobFunctions': job_func,
        'linkmanList.0.relationship': link0,
        'linkmanList.1.relationship': link1,
        'maritalStatus': marital,
        'pictureInfo.0.faceScore': face_score,
        'province': province,
        'purpose': purpose,
        'resideFunctions': reside,
        'term': term,
        'deviceInfo.gpsLatitude': lat,
        'deviceInfo.gpsLongitude': lon,
        'deviceInfo.osType': os_type,
        'companyInfo.companyName': company_name,
        'idInfo.gender': gender,
        'idInfo.nation': nation,
        'deviceInfo.isCrossDomain': is_cross_domain,
    }
    input_df = preprocess_input(input_dict)
    st.session_state.last_input_df = input_df
    st.session_state.last_model_choice = model_choice

    with st.spinner("正在预测并计算SHAP值..."):
        # 准备输入数据（是否需要缩放取决于模型）
        input_scaled = input_df.copy()
        if NUM_COLS:
            input_scaled[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

        model = models[model_choice]
        if model_choice == '逻辑回归 (改进)':
            X_input = input_scaled
        else:
            X_input = input_df

        proba = model.predict_proba(X_input)[0][1]
        pred_class = (proba >= 0.5).astype(int)
        st.session_state.last_pred_proba = proba
        st.session_state.last_pred_class = pred_class

        # SHAP 解释
        try:
            if model_choice == '逻辑回归 (改进)':
                base_path = os.path.dirname(__file__)
                data_path = os.path.join(base_path, 'data_now')
                try:
                    X_train_bg = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
                    background = X_train_bg.sample(n=100, random_state=42)
                except:
                    background = X_input.iloc[:1]
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_input)
                base_value = explainer.expected_value
                shap_values_single = shap_values[0]
                shap_values_single = np.clip(shap_values_single, -10, 10)
                exp = shap.Explanation(values=shap_values_single,
                                        base_values=base_value,
                                        data=X_input.iloc[0].values,
                                        feature_names=FEATURE_NAMES)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, show=False, max_display=15)
                st.session_state.last_shap_fig = fig
                plt.close()
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                if shap_values.ndim == 2:
                    shap_values_single = shap_values[0]
                else:
                    shap_values_single = shap_values
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_value = explainer.expected_value[1]
                else:
                    base_value = explainer.expected_value
                exp = shap.Explanation(values=shap_values_single,
                                        base_values=base_value,
                                        data=X_input.iloc[0].values,
                                        feature_names=FEATURE_NAMES)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, show=False, max_display=15)
                st.session_state.last_shap_fig = fig
                plt.close()
        except Exception as e:
            st.error(f"SHAP解释失败: {e}")
            st.session_state.last_shap_fig = None
            shap_values_single = None

        # 提取重要特征用于生成解释和保存用户优势
        if shap_values_single is not None:
            try:
                feature_values = X_input.iloc[0].values
                contributions = list(zip(FEATURE_NAMES, feature_values, shap_values_single))
                contributions.sort(key=lambda x: x[2], reverse=True)
                pos_features = [(name, val, shap) for name, val, shap in contributions if shap > 0]
                neg_features = [(name, val, shap) for name, val, shap in contributions if shap < 0]
                st.session_state.pred_explanation = generate_prediction_explanation(
                    proba, pos_features, neg_features, model_choice
                )
                # 保存用户优势特征（前3个正向特征，用于最佳决策理由）
                if pos_features:
                    adv_list = [f"{name}为{val:.1f}" for name, val, _ in pos_features[:3]]
                    st.session_state.last_user_advantages = "、".join(adv_list)
                else:
                    st.session_state.last_user_advantages = "无明显优势特征"
            except Exception as e:
                st.session_state.pred_explanation = f"无法生成解释：{e}"
                st.session_state.last_user_advantages = ""
        else:
            st.session_state.pred_explanation = "（无法获取SHAP值，解释不可用）"
            st.session_state.last_user_advantages = ""

        # 预测完成后立即 rerun 以显示结果（可选，但为了流畅可以不加）
        st.rerun()

# ========== 预测结果显示区域（独立于按钮） ==========
if st.session_state.last_pred_proba is not None:
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("通过概率", f"{st.session_state.last_pred_proba:.2%}",
                  delta=f"预测结果: {' 通过' if st.session_state.last_pred_class else ' 拒绝'}")
    with col_res2:
        st.caption(f"使用模型: {st.session_state.last_model_choice}")

    

    if st.session_state.last_shap_fig is not None:
        st.subheader(" 决策原因分析 (SHAP)")
        st.pyplot(st.session_state.last_shap_fig)
    else:
        st.subheader(" 特征重要性 (替代分析)")
        if st.session_state.last_model_choice == '逻辑回归 (改进)':
            model = models[st.session_state.last_model_choice]
            coef = model.coef_[0]
            feature_imp = pd.DataFrame({'特征': FEATURE_NAMES, '系数': coef})\
                           .assign(abs_coef=np.abs(coef)).sort_values('abs_coef', ascending=False).head(20)
            st.bar_chart(feature_imp.set_index('特征')['系数'])
            st.caption("正系数增加通过概率，负系数降低通过概率")
        elif hasattr(models[st.session_state.last_model_choice], 'feature_importances_'):
            model = models[st.session_state.last_model_choice]
            imp = model.feature_importances_
            feature_imp = pd.DataFrame({'特征': FEATURE_NAMES, '重要性': imp})\
                           .sort_values('重要性', ascending=False).head(20)
            st.bar_chart(feature_imp.set_index('特征')['重要性'])

if st.session_state.pred_explanation:
    st.caption(f" **简单解释**：{st.session_state.pred_explanation}")

st.markdown("---")

# ========== 第三部分：最佳决策推荐 ==========
st.markdown('<div class="section-title"><span>③</span> 最佳决策推荐</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">根据当前申请信息，计算不同借贷公司的通过概率，并为您推荐最匹配的公司。</div>', unsafe_allow_html=True)

if st.button(" 最佳决策", use_container_width=True, type="primary"):
    if st.session_state.last_input_df is None:
        st.warning("请先进行预测，再使用最佳决策功能")
    else:
        with st.spinner("正在计算各公司通过概率..."):
            input_df = st.session_state.last_input_df
            model_choice = st.session_state.last_model_choice
            model = models[model_choice]

            companies = label_encoders['partner_code'].classes_
            results = []
            for company in companies:
                temp_df = input_df.copy()
                temp_df['partner_code'] = label_encoders['partner_code'].transform([company])[0]
                if model_choice == '逻辑回归 (改进)':
                    temp_scaled = temp_df.copy()
                    if NUM_COLS:
                        temp_scaled[NUM_COLS] = scaler.transform(temp_df[NUM_COLS])
                    prob = model.predict_proba(temp_scaled)[0][1]
                else:
                    prob = model.predict_proba(temp_df)[0][1]
                results.append((company, prob))

            results_df = pd.DataFrame(results, columns=['公司', '通过概率']).sort_values('通过概率', ascending=False)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            best_company = results_df.iloc[0]['公司']
            best_prob = results_df.iloc[0]['通过概率']
            top3 = [(row['公司'], row['通过概率']) for _, row in results_df.head(3).iterrows()]
            # 传入 user_advantages
            st.session_state.best_reason = generate_best_decision_reason(
                best_company, best_prob, top3[1:], st.session_state.last_user_advantages
            )
            st.success(f" 最佳推荐：**{best_company}**，通过概率 **{best_prob:.2%}**")
            if st.session_state.best_reason:
                st.caption(f" 理由：{st.session_state.best_reason}")

# 页脚提示
# st.markdown('<div class="small-note">* 数据仅作界面示意，所有计算在本地完成</div>', unsafe_allow_html=True)