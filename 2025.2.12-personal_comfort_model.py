import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io  # 用于图形下载

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl'),
    'DT': joblib.load('dt_model.pkl'),
    'RF': joblib.load('rf_model.pkl')
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
    min_temp, max_temp = climate_temp_ranges[climate_zone_code][season_map[season_code]]

    if "Manual" in input_mode:
        default_temp = np.clip(15.0, min_temp, max_temp)
        Mean_Daily_Outdoor_Temperature = st.number_input(
            "Mean Daily Outdoor Temperature",
            min_value=float(min_temp),
            max_value=float(max_temp),
            value=default_temp,
            step=0.5,
            format="%.1f",
            help=f"Effective range of current climate zones：{min_temp}°C ~ {max_temp}°C"
        )
    else:
        st.info(f"Outdoor temperature generation range：{min_temp}°C ~ {max_temp}°C")
    
    # ===== 新增：图表颜色设置 =====
    st.sidebar.header("🎨 Chart Color Settings")
    with st.sidebar.expander("Pie Chart Colors", expanded=False):
        pie_color_0 = st.color_picker("Color for 'No change'", "#99ff99")
        pie_color_1 = st.color_picker("Color for 'Warmer'", "#ff9999")
        pie_color_2 = st.color_picker("Color for 'Cooler'", "#66b3ff")
    with st.sidebar.expander("Scatter Plot Colors", expanded=False):
        scatter_color = st.color_picker("Scatter point color", "#000000")
        vline_color_min = st.color_picker("Vertical line color for min temperature", "#0000FF")
        vline_color_max = st.color_picker("Vertical line color for max temperature", "#FF0000")
    with st.sidebar.expander("Logistic Regression Curve Colors", expanded=False):
        lr_color_0 = st.color_picker("Color for Thermal preference 0", "#0000FF")
        lr_color_1 = st.color_picker("Color for Thermal preference 1", "#FF0000")
        lr_color_2 = st.color_picker("Color for Thermal preference 2", "#008000")

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
        # 手动输入模式，允许用户输入多行数据
        st.sidebar.subheader("Enter multiple data rows:")
        data_rows = []
        for i in range(st.sidebar.number_input("Number of rows", min_value=1, value=1)):
            with st.sidebar.expander(f"Row {i+1}"):
                indoor_temp = st.number_input("Indoor Air Temperature", 10.0, 40.0, 25.0, key=f"temp_{i}")
                indoor_humidity = st.number_input("Indoor Relative Humidity", 0.0, 100.0, 50.0, key=f"humidity_{i}")
                indoor_velocity = st.number_input("Indoor Air Velocity", 0.0, 5.0, 0.1, key=f"velocity_{i}")
                data_rows.append([indoor_temp, indoor_humidity, indoor_velocity, Mean_Daily_Outdoor_Temperature])
        
        env_params = pd.DataFrame(data_rows, columns=['Indoor Air Temperature', 'Indoor Relative Humidity', 'Indoor Air Velocity', 'Mean Daily Outdoor Temperature'])
    else:
        n_samples = int(input_mode.split("(")[1].replace(")", ""))
        np.random.seed(42)
        env_params = {
            'Indoor Air Temperature': np.round(np.random.uniform(10, 40, n_samples), 1).tolist(),
            'Indoor Relative Humidity': np.round(np.random.uniform(30, 80, n_samples), 1).tolist(),
            'Indoor Air Velocity': np.round(np.random.uniform(0, 1.5, n_samples), 2).tolist(),
            'Mean Daily Outdoor Temperature': np.round(np.random.uniform(min_temp, max_temp, n_samples), 1).tolist()
        }
        env_params = pd.DataFrame(env_params)

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
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if set(df.columns) != set(feature_order):
        missing_columns = set(feature_order) - set(df.columns)
        raise ValueError(f"Missing in the data box：{missing_columns}")
    return df[feature_order]

# ================= 主界面显示模块 =================
st.title("🏢 Intelligent Prediction System for Building Thermal Comfort")
df = generate_data()

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
        # 对输入数据进行归一化处理
        scaled_df = scaler.transform(df)
        
        with st.spinner("Predictions are in progress, please wait..."):
            predictions = model.predict(scaled_df)
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

        results_df = df.copy()
        results_df["Projected results"] = predictions
        comfort_mapping = {
            0: "No change",
            1: "Warmer",
            2: "Cooler"
        }
        results_df["Comfort Evaluation"] = results_df["Projected results"].map(comfort_mapping)

        with st.expander("📊 View detailed forecast results", expanded=True):
            def highlight_tp(val):
                colors = {0: '#e6ffe6', 1: '#ffe6e6', 2: '#e6f3ff'}
                return f'background-color: {colors.get(val, "")}'
            styled_df = results_df.style.applymap(highlight_tp, subset=["Projected results"])
            st.dataframe(styled_df, height=300)

        st.subheader("📈 Analyzing Charts")
        col1, col2 = st.columns(2)
        
        # =========== 图形1：预测结果分布饼图 ===========
        with col1:
            fig1 = plt.figure(figsize=(8, 6))
            results_df["Comfort Evaluation"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=[pie_color_0, pie_color_1, pie_color_2],
                startangle=90,
                textprops={"fontsize": 12}
            )
            plt.title("Distribution of forecast results", fontsize=14)
            plt.ylabel("", fontsize=12)
            st.pyplot(fig1)
            
            # 下载饼图
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png')
            buf1.seek(0)
            st.download_button(
                label="Download Pie Chart",
                data=buf1,
                file_name="pie_chart.png",
                mime="image/png"
            )

        # =========== 图形2：温度-舒适度散点图 ===========
        with col2:
            fig2 = plt.figure(figsize=(8, 8))
            plt.scatter(
                results_df["Indoor Air Temperature"],
                results_df["Projected results"],
                c=scatter_color,
                alpha=0.7
            )
            zero_projected_results = results_df[results_df["Projected results"] == 0]
            if not zero_projected_results.empty:
                min_temp_at_zero = zero_projected_results["Indoor Air Temperature"].min()
                max_temp_at_zero = zero_projected_results["Indoor Air Temperature"].max()
                #plt.axvline(x=min_temp_at_zero, color=vline_color_min, linestyle=':', 
                            #label=f'Min Temp at Zero ({min_temp_at_zero:.2f}°C)')
                #plt.axvline(x=max_temp_at_zero, color=vline_color_max, linestyle=':', 
                            #label=f'Max Temp at Zero ({max_temp_at_zero:.2f}°C)')
            plt.legend()
            plt.title("Mapping of indoor air temperatures to predicted thermal preferences", fontsize=14)
            plt.xlabel("Indoor Air Temperature", fontsize=12)
            plt.ylabel("Thermal preference", fontsize=12)
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)
            
            # 下载散点图
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png')
            buf2.seek(0)
            st.download_button(
                label="Download Scatter Plot",
                data=buf2,
                file_name="scatter_plot.png",
                mime="image/png"
            )

        # 下载预测结果数据
        st.download_button(
            label="Download full forecast results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

        # ----------------- 新增：多项逻辑回归曲线及参数显示 -----------------
        with st.expander("📈 Multinomial Logistic Regression Curves", expanded=True):
            # 使用“Indoor Air Temperature”作为唯一特征构造多项逻辑回归模型
            X_multi = results_df["Indoor Air Temperature"].values.reshape(-1, 1)
            y_multi = results_df["Projected results"].values
            lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            lr_multi.fit(X_multi, y_multi)
    
        # 显示每一条回归曲线的参数和回归公式
        st.markdown("### Regression curve parameters and regression equation")
        intercepts = lr_multi.intercept_
        coefs = lr_multi.coef_  # shape (3, 1)
        
        # 遍历 3 个类别
        for idx in range(len(intercepts)):
            intercept = intercepts[idx]
            coef = coefs[idx][0]
            st.markdown(f"**Thermal preference {idx} （{comfort_mapping[idx]}）**")
            st.write(f"Intercept (β₀): {intercept:.4f}")
            st.write(f"Temperature coefficient (β₁): {coef:.4f}")
            st.markdown(
                f"**Regression equation:** $$p_{{{idx}}}(x)=\\frac{{\\exp({intercept:.4f}+{coef:.4f}x)}}{{\\exp({intercepts[0]:.4f}+{coefs[0][0]:.4f}x)+\\exp({intercepts[1]:.4f}+{coefs[1][0]:.4f}x)+\\exp({intercepts[2]:.4f}+{coefs[2][0]:.4f}x)}}$$"
            )
    
        # 选择是否显示每条逻辑回归曲线
        show_lr_0 = st.checkbox("Show Thermal preference 0 curve", value=True)
        show_lr_1 = st.checkbox("Show Thermal preference 1 curve", value=True)
        show_lr_2 = st.checkbox("Show Thermal preference 2 curve", value=True)
    
        # 绘制多项逻辑回归概率曲线
        temp_range_multi = np.linspace(results_df["Indoor Air Temperature"].min(),
                                       results_df["Indoor Air Temperature"].max(),
                                       1000).reshape(-1, 1)
        proba_multi = lr_multi.predict_proba(temp_range_multi)
    
        fig_multi, ax_multi = plt.subplots(figsize=(10, 6))
    
        if show_lr_0:
            ax_multi.plot(temp_range_multi, proba_multi[:, 0], label="Thermal preference 0", 
                      color=lr_color_0, linewidth=2)
        if show_lr_1:
            ax_multi.plot(temp_range_multi, proba_multi[:, 1], label="Thermal preference 1", 
                      color=lr_color_1, linewidth=2)
        if show_lr_2:
            ax_multi.plot(temp_range_multi, proba_multi[:, 2], label="Thermal preference 2", 
                      color=lr_color_2, linewidth=2)
        
        ax_multi.set_xlabel("Indoor Air Temperature (°C)", fontsize=12)
        ax_multi.set_ylabel("Predicted Probability", fontsize=12)
        ax_multi.set_title("Multinomial Logistic Regression Curves for Thermal Preference", fontsize=14)
        ax_multi.legend()
        ax_multi.grid(linestyle="--", alpha=0.3)
        st.pyplot(fig_multi)
    
        # 下载逻辑回归曲线图
        buf3 = io.BytesIO()
        fig_multi.savefig(buf3, format='png')
        buf3.seek(0)
        st.download_button(
            label="Download Logistic Regression Curve",
            data=buf3,
            file_name="logistic_regression_curve.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")

