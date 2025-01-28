import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# Load the models
LightGBM = joblib.load('lgbm_model.pkl')
GBM = joblib.load('gbm_model.pkl')
XGBoost = joblib.load('xgb_model.pkl')
RF = joblib.load('rf_model.pkl')
ET = joblib.load('et_model.pkl')
KNN = joblib.load('knn_model.pkl')
SVM = joblib.load('svm_model.pkl')
DT = joblib.load('dt_model.pkl')
ANN = joblib.load('ann_model.pkl')

# Model dictionary
models = {
    'LightGBM': LightGBM,
    'GBM': GBM,
    'XGBoost': XGBoost,
    'RF': RF,
    'ET': ET,
    'KNN': KNN,
    'SVM': SVM,
    'DT': DT,
    'ANN': ANN
}

# 标题
st.title("Thermal Comfort Prediction App")

# ---- 侧边栏输入设计 ----
with st.sidebar:
    st.header("Input Parameters")

    # ===== 第一层级: Basic Identifiers =====
    st.subheader("1. Basic Identifiers")
    
    # Season
    season = st.selectbox(
        "Season",
        ["Winter (0)", "Summer (1)", "Transition (2)"],
        index=0
    )
    season_code = int(season.split("(")[1].replace(")", ""))  # 提取编码
    
    # Climate Zone
    climate_zone = st.selectbox(
        "Climate Zone",
        ["Sever cold zone (0)", "Cold zone (1)", "Hot summer & cold winter (2)",
         "Hot summer & warm winter (3)", "Mild zones (4)"],
        index=0
    )
    climate_code = int(climate_zone.split("(")[1].replace(")", ""))

    # ===== 第二层级: Building Information =====
    st.subheader("2. Building Information")
    
    # Building Type
    building_type = st.selectbox(
        "Building Type",
        ["Dormitory (0)", "Educational (1)", "Office (2)", "Residential (3)"],
        index=0
    )
    building_code = int(building_type.split("(")[1].replace(")", ""))
    
    # Operation Mode (注意：Split air conditioner 和 self-heating 编码重复问题已修复)
    operation_mode = st.selectbox(
        "Building Operation Mode",
        ["Air conditioning heating (0)", "Ceiling capillary heating (1)", 
         "Cold radiation ceiling cooling (2)", "Convection cooling (3)",
         "Convection heating (4)", "Furnace heating (5)", 
         "Naturally Ventilated (6)", "Others (7)",
         "Radiant floor heating (8)", "Radiator heating (9)",
         "Self-heating (10)", "Split air conditioner (11)"],
        index=0
    )
    operation_code = int(operation_mode.split("(")[1].replace(")", ""))

    # ===== 第三层级: Subject's Personal Information =====
    st.subheader("3. Subject's Personal Information")
    
    sex = st.radio("Sex", ["Female (0)", "Male (1)"], index=0)
    sex_code = int(sex.split("(")[1].replace(")", ""))
    
    age_group = st.selectbox(
        "Age Group",
        ["<18 (0)", "18-30 (1)", "31-40 (2)", 
         "41-50 (3)", "51-60 (4)", ">61 (5)"],
        index=1
    )
    age_code = int(age_group.split("(")[1].replace(")", ""))
    
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=65)

    # ===== 第四层级: Thermal Comfort Information =====
    st.subheader("4. Subjective Thermal Comfort")
    
    clothing = st.number_input("Clothing Insulation (clo)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    metabolic = st.number_input("Metabolic Rate (met)", min_value=0.5, max_value=4.0, value=1.2, step=0.1)

    # ===== 第五层级: Indoor Physical Parameters =====
    st.subheader("5. Indoor Physical Parameters")
    
    # 输入模式选择
    input_mode = st.radio(
        "Input Mode", 
        ["Manual Input", "Random Generate (30)", "Random Generate (50)", "Random Generate (100)"]
    )
    
    # 根据模式生成数据
    if "Manual" in input_mode:
        temp = st.number_input("Indoor Air Temperature (°C)", min_value=10.0, max_value=40.0, value=25.0)
        humidity = st.number_input("Indoor Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        velocity = st.number_input("Indoor Air Velocity (m/s)", min_value=0.0, max_value=5.0, value=0.1)
    else:
        n_samples = int(input_mode.split("(")[1].replace(")", ""))
        np.random.seed(42)
        temp = np.round(np.random.uniform(18, 32, n_samples), 1)
        humidity = np.round(np.random.uniform(30, 80, n_samples), 1)
        velocity = np.round(np.random.uniform(0, 1.5, n_samples), 2)

# ---- 主页面显示结果 ----
st.header("Input Summary")

# 构建 DataFrame
data = {
    "Season": season_code,
    "Climate Zone": climate_code,
    "Building Type": building_code,
    "Operation Mode": operation_code,
    "Sex": sex_code,
    "Age Group": age_code,
    "Height (cm)": height,
    "Weight (kg)": weight,
    "Clothing (clo)": clothing,
    "Metabolic (met)": metabolic
}

if "Manual" in input_mode:
    data.update({
        "Temperature (°C)": temp,
        "Humidity (%)": humidity,
        "Velocity (m/s)": velocity
    })
    df = pd.DataFrame([data])
else:
    df = pd.DataFrame({
        "Temperature (°C)": temp,
        "Humidity (%)": humidity,
        "Velocity (m/s)": velocity
    })
    # 其他参数重复填充
    for key, value in data.items():
        df[key] = value

# 显示数据
st.dataframe(df)

# 添加下载按钮
st.download_button(
    label="Download Data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='thermal_comfort_data.csv'
)

# ---- 模型预测部分（示例） ----
if st.button("Run Prediction"):
    st.subheader("Prediction Results")
    # 此处添加模型预测代码
if st.button("预测"):
    model = models[selected_model]
    prediction = model.predict(input_features)  # 使用选定的模型进行预测
    
    st.subheader(f"Prediction results using the {selected_model} model")
    st.write(f"Predicted TP: {prediction[0]}")
    

# ================= TP值预测模块 =================
def plot_tp_temperature(samples, predictions):
    """生成温度-TP值关系图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义颜色和标签映射
    color_map = {0: 'gray', 1: 'red', 2: 'blue'}
    label_map = {0: 'No change', 1: 'Warmer', 2: 'Cooler'}
    
    # 绘制散点图
    for tp in [0, 1, 2]:
        mask = predictions == tp
        ax.scatter(
            samples['temperature'][mask],
            predictions[mask],
            c=color_map[tp],
            label=label_map[tp],
            alpha=0.7
        )
    
    # 设置图形属性
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Thermal Preference", fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(label_map.values())
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(title="TP Value")
    plt.tight_layout()
    return fig

# ================= 在预测逻辑中添加 =================
if st.button("开始多模型预测"):
    # ...（保留原有预测逻辑）
    
    # === 新增TP值预测可视化 ===
    st.subheader("🌡 温度-TP值关系分析")
    
    # 生成温度测试范围（示例用模型1预测）
    model = models['LightGBM']  # 任选一个模型
    temp_range = np.linspace(18, 32, 50)
    test_samples = samples.iloc[0:1].copy()
    
    tp_predictions = []
    for temp in temp_range:
        test_samples['temperature'] = temp
        pred = model.predict(test_samples)[0]
        tp_predictions.append(pred)
    
    # 绘制动态图表
    fig = plt.figure(figsize=(10, 4))
    plt.plot(temp_range, tp_predictions, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Predicted TP Value")
    plt.yticks([0, 1, 2], ['No change (0)', 'Warmer (1)', 'Cooler (2)'])
    plt.grid(linestyle='--', alpha=0.5)
    plt.title("Thermal Preference Prediction vs Temperature")
    st.pyplot(fig)

# ================= 在舒适度建模后添加 =================
if st.button("生成舒适度模型"):
    # ...（保留原有建模逻辑）
    
    # === 新增分类边界可视化 ===
    st.subheader("🧊 热偏好分类边界")
    
    # 生成网格数据
    x_temp = np.linspace(18, 32, 100)
    y_humidity = np.linspace(30, 70, 100)
    xx, yy = np.meshgrid(x_temp, y_humidity)
    
    # 预测分类结果（示例固定风速0.1m/s）
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 0.1)]).reshape(xx.shape)
    
    # 绘制等高线图
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], 
                         colors=['gray', 'red', 'blue'])
    ax.scatter(samples['temperature'], samples['humidity'], 
              c=samples['velocity'], cmap='viridis', s=50)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    plt.colorbar(contour, label="TP Value", 
                 ticks=[0, 1, 2]).set_ticklabels(['No change', 'Warmer', 'Cooler'])
    st.pyplot(fig)