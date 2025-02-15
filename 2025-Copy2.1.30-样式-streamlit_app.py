import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'GBM': joblib.load('gbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl')
}

# 页面配置
st.set_page_config(
    page_title="热舒适度预测系统",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙️ 参数输入面板")

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
         "Hot summer and warm winter zone  (3)", "Mild zone (4)"],
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
            ["<18 (0)", "18-30 (1)", "31-40 (2)", 
             "41-50 (3)", "51-60 (4)", ">61 (5)"],
            index=1
        )
    
    col3, col4 = st.columns(2)
    with col3:
        Height = st.number_input("Height", 100, 250, 170)
    with col4:
        Weight = st.number_input("Weight", 30, 150, 65)

    # 第四层级：Subjective Thermal Comfort Information
    st.subheader("4. Subjective Thermal Comfort Information")
    Clothing_Insulation = st.number_input("Clothing Insulation", 0.0, 2.0, 1.0, 0.1)
    Metabolic_Rate = st.number_input("Metabolic Rate", 0.5, 4.0, 1.2, 0.1)

    # 第五层级：Indoor Physical Parameters
    st.subheader("5. Indoor Physical Parameters")
    input_mode = st.radio(
        "Input pattern", 
        ["Manual input", "Randomly generate (30)", "Randomly generate (50)", "Randomly generate (100)"]
    )

    # 气候分区温度范围
    climate_code = int(Climate_Zone.split("(")[1].replace(")", ""))
    temp_ranges = {
        0: (-20, 5),    # 严寒地区
        1: (-10, 10),   # 寒冷地区
        2: (0, 25),     # 夏热冬冷
        3: (10, 35),    # 夏热冬暖
        4: (5, 30)      # 温和地区
    }
    min_temp, max_temp = temp_ranges[climate_code]

    if "Manual input" in input_mode:
        # 自动计算合理默认值
        default_temp = np.clip(15.0, min_temp, max_temp)  # 确保默认值在有效范围内
        
        outdoor_temp = st.number_input(
            "Mean Daily Outdoor Temperature",
            min_value=float(min_temp),  # 确保转换为float
            max_value=float(max_temp),
            value=default_temp,
            step=0.5,
            format="%.1f",
            help=f"当前气候分区有效范围：{min_temp}°C ~ {max_temp}°C"
        )
    else:
        st.info(f"室外温度生成范围：{min_temp}°C ~ {max_temp}°C")

# ================= 修改后的数据处理模块 =================
def generate_data():
    """生成与训练特征严格一致的数据框"""
    # 解析编码值（严格匹配训练特征名称）
    codes = {
        # 确保键名与训练特征完全一致（参考训练数据列名）
        'Season': int(Season.split("(")[1].replace(")", "")),
        'Climate_Zone': int(Climate_Zone.split("(")[1].replace(")", "")),  # 下划线命名
        'Building_Type': int(Building_Type.split("(")[1].replace(")", "")),
        'Building_Operation_Mode': int(Building_Operation_Mode.split("(")[1].replace(")", "")),
        'Sex': int(Sex.split("(")[1].replace(")", "")),
        'Age': int(Age.split("(")[1].replace(")", "")),
        'Height': Height,
        'Weight': Weight,
        'Clothing_Insulation': Clothing_Insulation,  # 匹配训练特征命名
        'Metabolic_Rate': Metabolic_Rate
    }

    # 生成环境参数（列名与训练数据严格一致）
    if "Manual input" in input_mode:
        env_params = {
            'Indoor_Air_Temperature': st.number_input("Indoor Air Temperature", 10.0, 40.0, 25.0),
            'Indoor_Relative_Humidity': st.number_input("Indoor Relative Humidity", 0.0, 100.0, 50.0),
            'Indoor_Air_Velocity': st.number_input("Indoor Air Velocity", 0.0, 5.0, 0.1),
            'Outdoor_Temperature': outdoor_temp
        }
    else:
        n_samples = int(input_mode.split("(")[1].replace(")", ""))
        np.random.seed(42)
        env_params = {
            'Indoor_Air_Temperature': np.round(np.random.uniform(18, 32, n_samples), 1),
            'Indoor_Relative_Humidity': np.round(np.random.uniform(30, 80, n_samples), 1),
            'Indoor_Air_Velocity': np.round(np.random.uniform(0, 1.5, n_samples), 2),
            'Outdoor_Temperature': np.round(np.random.uniform(min_temp, max_temp, n_samples), 1)
        }

    # 构建数据框（确保列顺序与训练时完全一致）
    feature_order = [
        # 按训练数据实际列顺序排列（需根据训练数据调整）
        'Season',
        'Climate_Zone',
        'Building_Type',
        'Building_Operation_Mode',
        'Sex',
        'Age',
        'Height',
        'Weight',
        'Clothing_Insulation',
        'Metabolic_Rate',
        'Indoor_Air_Temperature',
        'Indoor_Relative_Humidity',
        'Indoor_Air_Velocity',
        'Outdoor_Temperature'
    ]
    
    return pd.DataFrame([{**codes, **env_params}])[feature_order]  # 强制排序


# ================= 主界面显示模块 =================
st.title("🏢 建筑热舒适度智能预测系统")
df = generate_data()

# 检查 df 是否为空
if df.empty:
    st.warning("生成的数据为空，请检查输入参数。")
else:
    # 输入数据展示
    with st.expander("📥 查看输入数据", expanded=True):
        numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
        styled_df = df.style.format(lambda x: "{:.1f}" if x.name in numeric_columns else x)
        st.dataframe(styled_df, height=300)
        st.download_button(
            label="下载输入数据",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='input_data.csv'
        )

# ================= 预测分析模块 =================
st.header("🔮 预测分析")
selected_model = st.selectbox("选择预测模型", list(models.keys()))

# NEW: 加载保存的标准化器和类别权重
scaler = joblib.load('scaler.pkl')  # 确保与训练时使用相同的scaler

# ================= 预测分析模块 =================
if st.button("开始预测"):
    try:
        model = models[selected_model]
        
        # 执行预测
        with st.spinner("预测进行中，请稍候..."):
            scaled_df = scaler.transform(df)  # 使用训练时的scaler
            
            # 统一预测逻辑
            predictions = model.predict(scaled_df)
            raw_proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None
            
        # 构建结果数据框
        results_df = df.copy()
        results_df["预测结果"] = predictions
        
        # 添加概率分析列
        if raw_proba is not None:
            results_df["无需改变概率"] = raw_proba[:, 0]
            results_df["希望更暖概率"] = raw_proba[:, 1]
            results_df["希望更凉概率"] = raw_proba[:, 2]

        # 显示预测结果
        with st.expander("📊 查看详细预测结果", expanded=True):
            def highlight_tp(val):
                colors = {0: '#e6f3ff', 1: '#ffe6e6', 2: '#e6ffe6'}
                return f'background-color: {colors.get(val, "")}'
            
            styled_df = results_df.style.applymap(highlight_tp, subset=["预测结果"])
            st.dataframe(styled_df, height=300)

        # 可视化分析
        st.subheader("📈 分析图表")
        col1, col2, col3 = st.columns(3)

        with col1:
            # 预测结果分布
            fig1 = plt.figure(figsize=(8, 6))
            results_df["舒适度评价"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=["#66b3ff", "#ff9999", "#99ff99"],
                startangle=90
            )
            plt.title("预测结果分布")
            plt.ylabel("")
            st.pyplot(fig1)

        with col2:
            # 温度-舒适度关系
            fig2 = plt.figure(figsize=(8, 6))
            plt.scatter(
                results_df["Temperature (°C)"],
                results_df["预测结果"],
                c=results_df["预测结果"],
                cmap="coolwarm",
                alpha=0.7
            )
            plt.colorbar(ticks=[0, 1, 2]).set_ticklabels(["无需改变", "希望更暖", "希望更凉"])
            plt.xlabel("温度 (°C)")
            plt.ylabel("热舒适偏好")
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)

        # 新增概率分布分析
        with col3:
            if raw_proba is not None:
                fig3 = plt.figure(figsize=(8, 6))
                plt.hist(raw_proba[:, 1], bins=20, 
                        color='#ff9999', alpha=0.7,
                        range=(0, 1))
                plt.title("'希望更暖'概率分布")
                plt.xlabel("预测概率")
                plt.ylabel("样本数量")
                plt.grid(linestyle="--", alpha=0.3)
                st.pyplot(fig3)

        # 添加阈值调节滑块（仅针对XGBoost）
        if selected_model == 'XGBoost':
            st.subheader("⚖️ 阈值调整（实验功能）")
            col4, col5, col6 = st.columns(3)
            with col4:
                thresh_0 = st.slider("无需改变阈值", 0.0, 1.0, 0.3, 0.05)
            with col5:
                thresh_1 = st.slider("希望更暖阈值", 0.0, 1.0, 0.2, 0.05)
            with col6:
                thresh_2 = st.slider("希望更凉阈值", 0.0, 1.0, 0.3, 0.05)
            
            if st.button("应用动态阈值"):
                adjusted_pred = []
                for prob in raw_proba:
                    if prob[1] >= thresh_1 and prob[1] == max(prob):
                        adjusted_pred.append(1)
                    else:
                        adjusted_pred.append(np.argmax(prob))
                results_df["预测结果"] = adjusted_pred
                st.experimental_rerun()

        # 下载结果
        st.download_button(
            label="下载完整预测结果",
            data=results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")