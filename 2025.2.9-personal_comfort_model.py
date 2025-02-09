import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl')
}

scaler = joblib.load('scaler.pkl')  # 加载训练时保存的归一化器

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
    
    # 使用提取的编码查找对应的温度范围
    min_temp, max_temp = climate_temp_ranges[climate_zone_code][season_map[season_code]]

    if "Manual" in input_mode:
        # 自动计算合理默认值
        default_temp = np.clip(15.0, min_temp, max_temp)  # 确保默认值在有效范围内
        
        Mean_Daily_Outdoor_Temperature = st.number_input(
            "Mean Daily Outdoor Temperature",
            min_value=float(min_temp),  # 确保转换为float
            max_value=float(max_temp),
            value=default_temp,
            step=0.5,
            format="%.1f",
            help=f"Effective range of current climate zones：{min_temp}°C ~ {max_temp}°C"
        )
    else:
        st.info(f"Outdoor temperature generation range：{min_temp}°C ~ {max_temp}°C")

# ================= 数据处理模块 =================
def generate_data():
    """Generate data frames that are strictly consistent with the training features"""
    # 解析编码值（严格匹配训练特征名称）
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
    
    # 生成环境参数
    if "Manual input" in input_mode:
        env_params = {
            'Indoor Air Temperature': [st.number_input("Indoor Air Temperature", 10.0, 40.0, 25.0)],
            'Indoor Relative Humidity': [st.number_input("Indoor Relative Humidity", 0.0, 100.0, 50.0)],
            'Indoor Air Velocity': [st.number_input("Indoor Air Velocity", 0.0, 5.0, 0.1)],
            'Mean Daily Outdoor Temperature': [Mean_Daily_Outdoor_Temperature]
        }
    else:
        n_samples = int(input_mode.split("(")[1].replace(")", ""))
        np.random.seed(42)
        env_params = {
            'Indoor Air Temperature': np.round(np.random.uniform(18, 35, n_samples), 1).tolist(),
            'Indoor Relative Humidity': np.round(np.random.uniform(30, 80, n_samples), 1).tolist(),
            'Indoor Air Velocity': np.round(np.random.uniform(0, 1.5, n_samples), 2).tolist(),
            'Mean Daily Outdoor Temperature': np.round(np.random.uniform(min_temp, max_temp, n_samples), 1).tolist()
        }

    # 构建数据框（确保列顺序与训练时完全一致）
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
    
    # 创建数据框
    df = pd.DataFrame({**codes, **env_params})
    
    # 确保所有列都是数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 检查 df 的列名是否与 feature_order 完全一致
    if set(df.columns) != set(feature_order):
        missing_columns = set(feature_order) - set(df.columns)
        raise ValueError(f"Missing in the data box：{missing_columns}")
        
    return df[feature_order]

# ================= 主界面显示模块 =================
st.title("🏢 Intelligent Prediction System for Building Thermal Comfort")
df = generate_data()

# 输入数据展示
with st.expander("📥 Viewing Input Data", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="Download input data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='input_data.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 Predictive analysis")
selected_model = st.selectbox("Selecting a Predictive Model", list(models.keys()))

if st.button("Start forecasting"):
    try:
        model = models[selected_model]
        
        # 对输入数据进行归一化处理（使用预先加载的scaler）
        scaled_df = scaler.transform(df)  # 直接转换，无需删除列
        
        # 执行预测
        with st.spinner("Predictions are in progress, please wait..."):
            predictions = model.predict(scaled_df)
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

        # 构建结果数据框
        results_df = df.copy()  # 使用原始数据框作为基础
        results_df["Projected results"] = predictions
        # 定义舒适度评价的映射关系
        comfort_mapping = {
            0: "No change",
            1: "Warmer",
            2: "Cooler"
        }
        # 使用 map 函数将预测结果映射为舒适度评价
        results_df["Comfort Evaluation"] = results_df["Projected results"].map(comfort_mapping)

        # 显示预测结果
        with st.expander("📊 View detailed forecast results", expanded=True):
            # 条件格式
            def highlight_tp(val):
                colors = {0: '#e6ffe6', 1: '#ffe6e6', 2: '#e6f3ff'}
                return f'background-color: {colors.get(val, "")}'
            
            styled_df = results_df.style.applymap(highlight_tp, subset=["Projected results"])
            st.dataframe(styled_df, height=300)

        # 可视化分析
        st.subheader("📈 Analyzing Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            # 预测结果分布
            fig1 = plt.figure(figsize=(8, 6))
            results_df["Comfort Evaluation"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=["#99ff99", "#ff9999", "#66b3ff"],
                startangle=90,
                textprops={"fontsize": 12}  # 设置饼图中的百分比字体大小
            )
            plt.title("Distribution of forecast results", fontsize=14)
            plt.ylabel("", fontsize=12)  # 设置y轴标签字体大小
            st.pyplot(fig1)

        with col2:
            # 温度-舒适度关系
            fig2 = plt.figure(figsize=(8, 8))
            plt.scatter(
                results_df["Indoor Air Temperature"],
                results_df["Projected results"],
                c='black',
                alpha=0.7
            )
            # 筛选出预测值为0的数据
            zero_projected_results = results_df[results_df["Projected results"] == 0]
    
            # 获取预测值为0时的 Indoor Air Temperature 的最大值和最小值
            if not zero_projected_results.empty:
                min_temp_at_zero = zero_projected_results["Indoor Air Temperature"].min()
                max_temp_at_zero = zero_projected_results["Indoor Air Temperature"].max()
        
            # 绘制两条竖向的点线
                plt.axvline(x=min_temp_at_zero, color='blue', linestyle=':', label=f'Min Temp at Zero ({min_temp_at_zero:.2f}°C)')
                plt.axvline(x=max_temp_at_zero, color='red', linestyle=':', label=f'Max Temp at Zero ({max_temp_at_zero:.2f}°C)')
    
            # 添加图例
            plt.legend()
            
            plt.title("Mapping of indoor air temperatures to predicted thermal preferences", fontsize=14)
            plt.xlabel("Indoor Air Temperature", fontsize=12)
            plt.ylabel("Thermal preference", fontsize=12)
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)

        # 下载结果
        st.download_button(
            label="Download full forecast results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

        # ================= 新增：建立逻辑回归曲线 =================
        with st.expander("📈 Logistic Regression Curves", expanded=True):
            # 使用室内空气温度作为自变量
            # 构造温度范围，用于绘制平滑的概率曲线
            temp_range = np.linspace(results_df["Indoor Air Temperature"].min(),
                                     results_df["Indoor Air Temperature"].max(),
                                     1000).reshape(-1, 1)
            
            # ----------------- Thermal preference 0 vs. 1 -----------------
            # 筛选出标签为0和1的数据
            subset_01 = results_df[results_df["Projected results"].isin([0, 1])]
            # 使用“Indoor Air Temperature”作为唯一特征
            X_01 = subset_01["Indoor Air Temperature"].values.reshape(-1, 1)
            y_01 = subset_01["Projected results"].values  # 标签为0或1
            
            # 建立逻辑回归模型
            lr_01 = LogisticRegression()
            lr_01.fit(X_01, y_01)
            # 对温度范围内的点预测类别1的概率
            proba_01 = lr_01.predict_proba(temp_range)[:, 1]  # 第1列为类别1的概率

            # ----------------- Thermal preference 0 vs. 2 -----------------
            # 筛选出标签为0和2的数据
            subset_02 = results_df[results_df["Projected results"].isin([0, 2])]
            X_02 = subset_02["Indoor Air Temperature"].values.reshape(-1, 1)
            y_02 = subset_02["Projected results"].values
            # 为了构造二分类模型，将标签 2 转换为 1，其余保持为0
            y_02_binary = np.where(y_02 == 2, 1, 0)
            
            lr_02 = LogisticRegression()
            lr_02.fit(X_02, y_02_binary)
            # 对温度范围内的点预测类别（原标签2）对应的概率
            proba_02 = lr_02.predict_proba(temp_range)[:, 1]
            
            # 绘制两条逻辑回归概率曲线
            fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
            ax_lr.plot(temp_range, proba_01, label="Logistic Regression (0 vs. 1)", color='blue')
            ax_lr.plot(temp_range, proba_02, label="Logistic Regression (0 vs. 2)", color='red', linestyle='--')
            ax_lr.set_xlabel("Indoor Air Temperature (°C)", fontsize=12)
            ax_lr.set_ylabel("Predicted Probability", fontsize=12)
            ax_lr.set_title("Logistic Regression Curves for Thermal Preference", fontsize=14)
            ax_lr.legend()
            st.pyplot(fig_lr)

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")