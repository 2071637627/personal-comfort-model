import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl')
}

scaler = joblib.load('scaler.pkl')

# 页面配置
st.set_page_config(
    page_title="Thermal comfort prediction system",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙ Parameter Input Panel")

    # 第一层级：Basic Identifiers
    st.subheader("1. Basic Identifiers")
    Season = st.selectbox(
        "Season",
        ["Winter Season (0)", "Summer Season (1)", "Transition Season (2)"],
        index=0
    )
    Climate_Zone = st.selectbox(
        "Climate Zone",
        ["Severe cold zone (0)", "Cold zone (1)", "Hot summer and cold winter zone (2)",
         "Hot summer and warm winter zone (3)", "Mild zone (4)"],
        index=0
    )

    # 第二层级：Building Information
    st.subheader("2. Building Information")
    Building_Type = st.selectbox(
        "Building Type",
        ["Dormitory (0)", "Educational (1)", "Office (2)", "Residential (3)"],
        index=0
    )
    Building_Operation_Mode = st.selectbox(
        "Building Operation Mode",
        ["Air conditioning heating (0)", "Ceiling capillary heating (1)",
         "Cold radiation ceiling cooling (2)", "Convection cooling (3)",
         "Convection heating (4)", "Furnace heating (5)",
         "Naturally Ventilated (6)", "Others (7)",
         "Radiant floor heating (8)", "Radiator heating (9)",
         "self-heating (10)", "Split air conditioner (11)"],
        index=0
    )

    # 第三层级：Subject's Personal Information
    st.subheader("3. Subject's Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        Sex = st.radio("Sex", ["Female (0)", "Male (1)"], index=0)
    with col2:
        Age = st.selectbox(
            "Age",
            ["<18 (0)", "18-30 (1)", "31-40 (2)", "41-50 (3)", "51-60 (4)", ">61 (5)"],
            index=1
        )
    
    col3, col4 = st.columns(2)
    with col3:
        Height = st.number_input("Height (cm)", 100, 250, 170)
    with col4:
        Weight = st.number_input("Weight (kg)", 30, 150, 65)

    # 第四层级：Subjective Thermal Comfort Information
    st.subheader("4. Subjective Thermal Comfort Information")
    Clothing_Insulation = st.number_input("Clothing Insulation (clo)", 0.0, 2.0, 1.0, 0.1)
    Metabolic_Rate = st.number_input("Metabolic Rate (met)", 0.5, 4.0, 1.2, 0.1)

    # 第五层级：Indoor Physical Parameters
    st.subheader("5. Indoor Physical Parameters")
    input_mode = st.radio(
        "Input pattern", 
        ["Manual input", "Randomly generate (30)", "Randomly generate (50)", "Randomly generate (100)"]
    )

    # 气候分区温度范围
    climate_zone_code = int(Climate_Zone.split('(')[1].split(')')[0])
    season_code = int(Season.split('(')[1].split(')')[0])
    climate_temp_ranges = {
        0: {"Winter": (-20, 5), "Summer": (15, 25), "Transition": (5, 15)},
        1: {"Winter": (-10, 10), "Summer": (20, 30), "Transition": (10, 20)},
        2: {"Winter": (0, 10), "Summer": (25, 35), "Transition": (10, 25)},
        3: {"Winter": (5, 15), "Summer": (30, 35), "Transition": (15, 30)},
        4: {"Winter": (5, 15), "Summer": (20, 30), "Transition": (10, 25)}
    }
    season_map = {0: "Winter", 1: "Summer", 2: "Transition"}
    min_temp, max_temp = climate_temp_ranges[climate_zone_code][season_map[season_code]]

# ================= 数据处理模块 =================
def generate_data():
    """Generate data frames that are strictly consistent with the training features"""
    codes = {
        'Season': int(Season.split("(")[1].replace(")", "")),
        'Climate Zone': int(Climate_Zone.split("(")[1].replace(")", "")),
        'Building Type': int(Building_Type.split("(")[1].replace(")", "")),
        'Building Operation Mode': int(Building_Operation_Mode.split("(")[1].replace(")", "")),
        'Sex': int(Sex.split("(")[1].replace(")", "")),
        'Age': int(Age.split("(")[1].replace(")", "")),
        'Height': Height,
        'Weight': Weight,
        'Clothing Insulation': Clothing_Insulation,
        'Metabolic Rate': Metabolic_Rate
    }
    
    if "Manual input" in input_mode:
        st.subheader("🌡️ Environmental Parameters Input")
        col1, col2 = st.columns(2)
        with col1:
            indoor_temp_input = st.text_input(
                "Indoor Air Temperature (℃, comma separated)", 
                "25.0, 26.0, 24.5",
                help=f"Valid range: 10.0-40.0℃"
            )
            indoor_humidity_input = st.text_input(
                "Indoor Relative Humidity (%RH, comma separated)",
                "50.0, 55.0, 60.0",
                help="Valid range: 0.0-100.0%"
            )
        with col2:
            indoor_air_velocity_input = st.text_input(
                "Indoor Air Velocity (m/s, comma separated)",
                "0.1, 0.2, 0.15",
                help="Valid range: 0.0-5.0m/s"
            )
            mean_outdoor_temp_input = st.text_input(
                f"Outdoor Temperature (℃, comma separated)",
                f"{min_temp+5:.1f}, {max_temp-5:.1f}, {(min_temp+max_temp)/2:.1f}",
                help=f"Climate range: {min_temp}~{max_temp}℃"
            )

        try:
            # 数据转换和校验
            env_params = {
                'Indoor Air Temperature': [float(x.strip()) for x in indoor_temp_input.split(',')],
                'Indoor Relative Humidity': [float(x.strip()) for x in indoor_humidity_input.split(',')],
                'Indoor Air Velocity': [float(x.strip()) for x in indoor_air_velocity_input.split(',')],
                'Mean Daily Outdoor Temperature': [float(x.strip()) for x in mean_outdoor_temp_input.split(',')]
            }
            
            # 校验数据长度一致性
            param_lengths = [len(v) for v in env_params.values()]
            if len(set(param_lengths)) > 1:
                raise ValueError("所有参数必须包含相同数量的数值")
                
            # 校验数值范围
            for temp in env_params['Indoor Air Temperature']:
                if not (10.0 <= temp <= 40.0):
                    raise ValueError(f"室内温度{temp}℃超出有效范围（10.0-40.0℃）")
                    
            for temp in env_params['Mean Daily Outdoor Temperature']:
                if not (min_temp <= temp <= max_temp):
                    raise ValueError(f"室外温度{temp}℃超出气候分区范围（{min_temp}-{max_temp}℃）")

            for rh in env_params['Indoor Relative Humidity']:
                if not (0.0 <= rh <= 100.0):
                    raise ValueError(f"相对湿度{rh}%超出有效范围（0-100%）")

            for vel in env_params['Indoor Air Velocity']:
                if not (0.0 <= vel <= 5.0):
                    raise ValueError(f"空气流速{vel}m/s超出有效范围（0.0-5.0m/s）")

        except Exception as e:
            st.error(f"输入数据错误: {str(e)}")
            st.stop()

    else:
        n_samples = int(input_mode.split("(")[1].replace(")", ""))
        np.random.seed(42)
        env_params = {
            'Indoor Air Temperature': np.round(np.random.uniform(10, 40, n_samples), 1).tolist(),
            'Indoor Relative Humidity': np.round(np.random.uniform(30, 80, n_samples), 1).tolist(),
            'Indoor Air Velocity': np.round(np.random.uniform(0, 1.5, n_samples), 2).tolist(),
            'Mean Daily Outdoor Temperature': np.round(np.random.uniform(min_temp, max_temp, n_samples), 1).tolist()
        }

    feature_order = [
        'Season',
        'Climate Zone',
        'Building Type',
        'Building Operation Mode',
        'Sex',
        'Age',
        'Height',
        'Weight',
        'Clothing Insulation',
        'Metabolic Rate',
        'Indoor Air Temperature',
        'Indoor Relative Humidity',
        'Indoor Air Velocity',
        'Mean Daily Outdoor Temperature'
    ]
    
    df = pd.DataFrame({**codes, **env_params})
    return df[feature_order]

# ================= 主界面显示模块 =================
st.title("🏢 Intelligent Prediction System for Building Thermal Comfort")
df = generate_data()

with st.expander("📥 View Input Data", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="Download Input Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='input_data.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 Predictive Analysis")
selected_model = st.selectbox("Select Prediction Model", list(models.keys()))

if st.button("Start Prediction"):
    try:
        with st.spinner("🔮 Predicting..."):
            # 数据预处理
            scaled_df = scaler.transform(df)
            
            # 模型预测
            model = models[selected_model]
            predictions = model.predict(scaled_df)
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

            # 构建结果数据框
            results_df = df.copy()
            results_df["Predicted Class"] = predictions
            results_df["Comfort Evaluation"] = results_df["Predicted Class"].map({
                0: "No change",
                1: "Warmer",
                2: "Cooler"
            })

            # 显示结果
            st.success("✅ Prediction Completed!")
            
            with st.expander("📊 View Detailed Results", expanded=True):
                def highlight_tp(val):
                    colors = {0: '#e6ffe6', 1: '#ffe6e6', 2: '#e6f3ff'}
                    return f'background-color: {colors.get(val, "")}'
                styled_df = results_df.style.applymap(highlight_tp, subset=["Predicted Class"])
                st.dataframe(styled_df, height=300)

            # 图表显示
            st.subheader("📈 Analysis Charts")
            col1, col2 = st.columns(2)
            
            # 饼图
            with col1:
                fig1 = plt.figure(figsize=(8, 6))
                results_df["Comfort Evaluation"].value_counts().plot.pie(
                    autopct="%1.1f%%",
                    colors=["#99ff99", "#ff9999", "#66b3ff"],
                    startangle=90
                )
                plt.title("Prediction Results Distribution")
                st.pyplot(fig1)

            # 散点图
            with col2:
                fig2 = plt.figure(figsize=(8, 6))
                plt.scatter(
                    results_df["Indoor Air Temperature"],
                    results_df["Predicted Class"],
                    c="#000000",
                    alpha=0.7
                )
                plt.title("Temperature vs Thermal Preference")
                plt.xlabel("Indoor Air Temperature (°C)")
                plt.ylabel("Thermal Preference")
                plt.grid(linestyle="--", alpha=0.3)
                st.pyplot(fig2)

            # 下载功能
            st.download_button(
                label="Download Full Results",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f'predictions_{selected_model}.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Possible causes:\n1. Invalid input data format\n2. Missing model files\n3. Feature mismatch")

# ================= 样式优化 =================
st.markdown("""
<style>
    .stNumberInput, .stTextInput, .stSelectbox, .stRadio {
        padding: 5px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 5px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)